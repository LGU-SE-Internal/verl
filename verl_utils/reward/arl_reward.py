"""ARL testing-based reward function for verl.

Creates fresh K8s pods per instance, applies the agent's patch,
runs tests, and returns binary reward (0.0 or 1.0).

Supports both R2E-Gym and SWE-Bench Verified datasets.
"""

import json
import logging
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# ── Matches R2E-Gym's DOCKER_PATH ──────────────────────────────────
DOCKER_PATH = (
    "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin"
    ":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)

MAX_CONCURRENT_PODS = int(os.environ.get("ARL_REWARD_CONCURRENCY", "16"))
REWARD_TIMEOUT = int(os.environ.get("ARL_REWARD_TIMEOUT", "300"))


# ── Parsers (ported from rllm/environments/swe/reward.py) ──────────


def parse_log_pytest(log: str | None) -> dict[str, str]:
    """Parse pytest 'short test summary info' section."""
    if log is None:
        return {}
    test_status_map = {}
    if "short test summary info" not in log:
        return test_status_map
    log = log.split("short test summary info")[1].strip()
    for line in log.split("\n"):
        if "PASSED" in line:
            test_name = ".".join(line.split("::")[1:])
            test_status_map[test_name] = "PASSED"
        elif "FAILED" in line:
            test_name = ".".join(line.split("::")[1:]).split(" - ")[0]
            test_status_map[test_name] = "FAILED"
        elif "ERROR" in line:
            try:
                test_name = ".".join(line.split("::")[1:])
            except IndexError:
                test_name = line
            test_name = test_name.split(" - ")[0]
            test_status_map[test_name] = "ERROR"
    return test_status_map


def decolor_dict_keys(key_dict: dict) -> dict:
    """Remove ANSI escape codes from dictionary keys."""
    decolor = lambda key: re.sub(r"\u001b\[\d+m", "", key)
    return {decolor(k): v for k, v in key_dict.items()}


# ── Session helpers ─────────────────────────────────────────────────


def _mirror_image(docker_image: str) -> str:
    registry = os.environ.get(
        "ARL_MIRROR_REGISTRY", "pair-diag-cn-guangzhou.cr.volces.com"
    )
    if not registry:
        return docker_image
    namespace = os.environ.get("ARL_MIRROR_NAMESPACE", "code")
    parts = docker_image.split("/", 1)
    image_path = parts[1] if len(parts) == 2 else docker_image
    return f"{registry}/{namespace}/{image_path}"


def _run_in_session(
    session, cmd: str, workdir: str, timeout: int, swebench_verified: bool = False
) -> tuple[str, str]:
    """Execute a command in the sandbox session. Returns (output, error_code_str)."""
    if swebench_verified:
        shell_cmd = (
            f"source /opt/miniconda3/bin/activate && "
            f"conda activate testbed && "
            f"{cmd}"
        )
        step = {
            "name": "reward_cmd",
            "command": ["bash", "-c", shell_cmd],
            "workDir": workdir,
            "timeout": timeout,
        }
    else:
        step = {
            "name": "reward_cmd",
            "command": ["bash", "-c", cmd],
            "env": {"PATH": DOCKER_PATH},
            "workDir": workdir,
            "timeout": timeout,
        }

    response = session.execute(steps=[step])
    result = response.results[0]
    output = result.output.stdout
    if result.output.stderr:
        output = (
            output + "\n" + result.output.stderr if output else result.output.stderr
        )
    exit_code = result.output.exit_code
    output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)

    if exit_code == 124:
        return f"The command took too long to execute (>{timeout}s)", "-1"
    if exit_code != 0:
        return output, f"Error: Exit code {exit_code}"
    return output, str(exit_code)


# ── R2E-Gym reward ─────────────────────────────────────────────────


def _calculate_reward_r2e(
    session, extra_info: dict, repo_path: str, alt_path: str, timeout: int
) -> float:
    """R2E-Gym reward via pytest output comparison."""
    output, _ = _run_in_session(
        session, f"bash {alt_path}/run_tests.sh", repo_path, timeout
    )
    parse = parse_log_pytest(output)
    parse = decolor_dict_keys(parse)

    expected_json = extra_info.get("expected_output_json", "")
    if not expected_json:
        # Fallback: read from container
        expected_json, _ = _run_in_session(
            session, f"cat {alt_path}/expected_test_output.json", repo_path, 30
        )

    try:
        expected: dict = json.loads(expected_json)
    except (json.JSONDecodeError, TypeError):
        logger.error("Failed to parse expected output JSON")
        return 0.0
    expected = decolor_dict_keys(expected)
    parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys())}
    expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys())}

    if len(parse) != len(expected):
        return 0.0
    for k in parse.keys():
        if not k:
            continue
        if k not in expected or parse[k] != expected[k]:
            return 0.0
    return 1.0


# ── SWE-Bench Verified reward ──────────────────────────────────────


def _calculate_reward_swebench(
    session, extra_info: dict, timeout: int
) -> float:
    """SWE-Bench Verified reward via swebench harness."""
    from swebench.harness.constants import (APPLY_PATCH_FAIL, FAIL_ONLY_REPOS,
                                            FAIL_TO_PASS, KEY_INSTANCE_ID,
                                            MAP_REPO_VERSION_TO_SPECS,
                                            PASS_TO_PASS, RESET_FAILED,
                                            TESTS_ERROR, TESTS_TIMEOUT,
                                            EvalType, ResolvedStatus)
    from swebench.harness.grading import (get_eval_tests_report,
                                          get_resolution_status)
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
    from swebench.harness.test_spec.test_spec import make_test_spec

    test_spec = make_test_spec(extra_info)
    out, _ = _run_in_session(
        session, "/run_tests.sh", "/testbed", timeout, swebench_verified=True
    )

    # Parse logs
    repo = test_spec.repo
    version = test_spec.version
    log_parser = MAP_REPO_TO_PARSER[repo]
    test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
    if isinstance(test_cmd, list):
        test_cmd = test_cmd[-1]

    bad_codes = [
        x
        for x in [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT]
        if x in out
    ]
    if bad_codes:
        logger.error(f"Bad code found in log: {bad_codes}")
        return 0.0

    content = out.split(test_cmd)[-1]
    eval_status_map = log_parser(content, test_spec)

    eval_ref = {
        KEY_INSTANCE_ID: test_spec.instance_id,
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }
    eval_type = (
        EvalType.FAIL_ONLY
        if test_spec.repo in FAIL_ONLY_REPOS
        else EvalType.PASS_AND_FAIL
    )
    report = get_eval_tests_report(eval_status_map, eval_ref, eval_type=eval_type)
    success = get_resolution_status(report) == ResolvedStatus.FULL.value
    return float(success)


# ── Single-instance reward worker ──────────────────────────────────


def _score_single_instance(
    patch: str, extra_info: dict
) -> float:
    """Create a pod, apply patch, run tests, return reward."""
    from arl import ManagedSession

    docker_image = extra_info.get("docker_image", "")
    if not docker_image:
        logger.error("No docker_image in extra_info")
        return 0.0

    image = _mirror_image(docker_image)
    swebench_verified = "swebench" in docker_image
    repo_path = "/testbed"
    alt_path = "/" if swebench_verified else "/root"

    gateway_url = os.environ.get("ARL_GATEWAY_URL", "http://localhost:8080")
    namespace = os.environ.get("ARL_REWARD_NAMESPACE", "default")
    experiment_id = os.environ.get("ARL_EXPERIMENT_ID", "reward")

    session = ManagedSession(
        image=image,
        experiment_id=experiment_id,
        namespace=namespace,
        gateway_url=gateway_url,
        timeout=REWARD_TIMEOUT + 60,
    )

    try:
        session.create_sandbox()

        # Apply patch
        if patch.strip():
            # Ensure patch ends with exactly one newline (git requires it).
            import base64
            if not patch.endswith("\n"):
                patch += "\n"
            b64 = base64.b64encode(patch.encode()).decode()
            write_out, write_err = _run_in_session(
                session,
                f"printf '%s' '{b64}' | base64 -d > /tmp/agent.patch",
                repo_path,
                30,
                swebench_verified=swebench_verified,
            )
            if "Error" in str(write_err):
                logger.warning(f"Failed to write patch file: {write_out}")
                return 0.0
            out, err = _run_in_session(
                session,
                "git apply /tmp/agent.patch",
                repo_path,
                30,
                swebench_verified=swebench_verified,
            )
            if "Error" in err:
                logger.warning(f"git apply failed: {out}")
                return 0.0

        # Run tests and compute reward
        if swebench_verified:
            return _calculate_reward_swebench(session, extra_info, REWARD_TIMEOUT)
        else:
            # Setup environment for R2E-Gym
            _run_in_session(
                session,
                f"ln -sf {repo_path}/.venv {alt_path}/.venv",
                repo_path,
                30,
            )
            _run_in_session(
                session,
                f"find {repo_path}/.venv/bin -type f -executable "
                f"-exec ln -sf {{}} {alt_path}/.local/bin/ \\;",
                repo_path,
                30,
            )
            # Move test artifacts out of testbed
            _run_in_session(
                session,
                f"[ -f {repo_path}/run_tests.sh ] && mv {repo_path}/run_tests.sh {alt_path}/run_tests.sh || true",
                repo_path,
                10,
            )
            _run_in_session(
                session,
                f"[ -d /r2e_tests ] && mv /r2e_tests {alt_path}/r2e_tests && ln -s {alt_path}/r2e_tests {repo_path}/r2e_tests || true",
                repo_path,
                10,
            )
            return _calculate_reward_r2e(
                session, extra_info, repo_path, alt_path, REWARD_TIMEOUT
            )

    except Exception as e:
        logger.error(f"ARL reward error: {e}\n{traceback.format_exc()}")
        return 0.0
    finally:
        try:
            session.delete_sandbox()
        except Exception:
            pass
        try:
            session.close()
        except Exception:
            pass


# ── verl reward function interface ─────────────────────────────────


def compute_score_arl(
    data_sources, solution_strs, ground_truths, extra_infos
) -> list[float]:
    """ARL testing-based reward. Creates pods, applies patches, runs tests.

    Signature matches verl custom_reward_function interface.
    """
    from verl_utils.reward.extract_answer import extract_patch

    patches = [extract_patch(sol) for sol in solution_strs]
    scores = [0.0] * len(patches)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PODS) as pool:
        futures = {}
        for idx, (patch, extra_info) in enumerate(zip(patches, extra_infos)):
            if not patch.strip():
                scores[idx] = -1.0  # convention: no patch = invalid
                continue
            future = pool.submit(_score_single_instance, patch, extra_info)
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                scores[idx] = future.result()
            except Exception as e:
                logger.error(f"Reward computation failed for index {idx}: {e}")
                scores[idx] = 0.0

    return scores


def compute_score_arl_clip(
    data_sources, solution_strs, ground_truths, extra_infos
) -> list[float]:
    """ARL reward with -1.0 clipped to 0.0 (matches compute_score_remote_clip)."""
    scores = compute_score_arl(data_sources, solution_strs, ground_truths, extra_infos)
    return [0.0 if s == -1.0 else s for s in scores]
