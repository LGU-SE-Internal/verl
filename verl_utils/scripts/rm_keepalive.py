#!/usr/bin/env python3
"""
RM Server Keep-Alive
Sends mock requests using the same requests-based logic as model_client.py,
so there's no shell ARG_MAX limit and payload can be arbitrarily large.

Usage:
    RM_SERVER_URL=http://host:8365/score python3 verl_utils/scripts/rm_keepalive.py
    # or via shell wrapper:
    bash verl_utils/scripts/rm_keepalive.sh
"""
import os
import sys
import time

import requests

# ---------------------------------------------------------------------------
# Config (overridable via env vars)
# ---------------------------------------------------------------------------
SERVER_URL      = os.environ.get("RM_SERVER_URL",          "http://localhost:8365/score")
INTERVAL        = int(os.environ.get("RM_KEEPALIVE_INTERVAL", "300"))   # seconds between rounds
NUM_BATCHES     = int(os.environ.get("RM_KEEPALIVE_BATCHES",  "100"))   # batches per request
BATCH_SIZE      = 4                                                       # patches per batch (fixed by server)

# ---------------------------------------------------------------------------
# Mock data — multi-file, multi-hunk Python patch (~5000 tokens per copy)
# Designed to survive get_pure_patch() filtering:
#   - targets .py files (not test/reproduce/new-file)
#   - contains real code changes (not whitespace/comments only)
# ---------------------------------------------------------------------------
MOCK_ISSUE = (
    "The DataProcessor class in processor.py has a critical bug in the batch processing "
    "pipeline. When processing large datasets with mixed types (numerical and categorical), "
    "the normalize_batch method incorrectly handles NaN values, causing downstream aggregation "
    "to silently produce wrong results. Additionally, the parallel execution mode in executor.py "
    "fails to properly synchronize shared state across worker threads, leading to race conditions "
    "when accumulating partial results. The connection pool in connection_manager.py also leaks "
    "connections under high concurrency because the cleanup routine does not account for "
    "connections that are checked out but idle beyond the timeout window. Please fix these three "
    "issues while maintaining backward compatibility with existing callers."
)

MOCK_PATCH = """\
diff --git a/src/processor.py b/src/processor.py
--- a/src/processor.py
+++ b/src/processor.py
@@ -15,45 +15,78 @@
 import numpy as np
 from typing import List, Dict, Any, Optional, Tuple, Union
 from dataclasses import dataclass, field
+from collections import defaultdict
+import warnings


 @dataclass
 class BatchConfig:
     batch_size: int = 256
     max_retries: int = 3
-    nan_strategy: str = "skip"
+    nan_strategy: str = "interpolate"
     dtype_map: Dict[str, str] = field(default_factory=dict)
+    nan_fill_value: Optional[float] = None
+    strict_type_check: bool = True


 class DataProcessor:
-    def __init__(self, config: Optional[BatchConfig] = None):
+    def __init__(self, config: Optional[BatchConfig] = None, validate: bool = True):
         self.config = config or BatchConfig()
         self._cache: Dict[str, np.ndarray] = {}
         self._type_registry: Dict[str, type] = {}
+        self._nan_counts: Dict[str, int] = defaultdict(int)
+        self._validate = validate

     def normalize_batch(
         self, data: List[Dict[str, Any]], columns: Optional[List[str]] = None
     ) -> np.ndarray:
         if not data:
             return np.array([])
-
+
         columns = columns or list(data[0].keys())
         result = np.zeros((len(data), len(columns)), dtype=np.float64)
-
+
         for col_idx, col_name in enumerate(columns):
             raw_values = [row.get(col_name) for row in data]
             col_type = self._infer_column_type(raw_values)
-
+
             if col_type == "numerical":
-                for row_idx, val in enumerate(raw_values):
-                    if val is not None:
-                        result[row_idx, col_idx] = float(val)
+                numeric_vals = np.array(
+                    [float(v) if v is not None and not self._is_nan(v) else np.nan for v in raw_values],
+                    dtype=np.float64,
+                )
+                nan_mask = np.isnan(numeric_vals)
+                nan_count = int(np.sum(nan_mask))
+
+                if nan_count > 0:
+                    self._nan_counts[col_name] += nan_count
+                    if self.config.nan_strategy == "interpolate":
+                        numeric_vals = self._interpolate_nans(numeric_vals, nan_mask)
+                    elif self.config.nan_strategy == "fill":
+                        fill_val = self.config.nan_fill_value if self.config.nan_fill_value is not None else 0.0
+                        numeric_vals[nan_mask] = fill_val
+                    elif self.config.nan_strategy == "drop":
+                        import warnings
+                        warnings.warn(
+                            f"Column '{col_name}' has {nan_count} NaN values that will be zeroed in output"
+                        )
+                        numeric_vals[nan_mask] = 0.0
                     else:
-                        if self.config.nan_strategy == "skip":
-                            result[row_idx, col_idx] = 0.0
-                        elif self.config.nan_strategy == "mean":
-                            result[row_idx, col_idx] = 0.0
+                        numeric_vals[nan_mask] = 0.0
+
+                col_mean = np.nanmean(numeric_vals)
+                col_std = np.nanstd(numeric_vals)
+                if col_std > 1e-10:
+                    result[:, col_idx] = (numeric_vals - col_mean) / col_std
+                else:
+                    result[:, col_idx] = numeric_vals - col_mean
+
             elif col_type == "categorical":
-                unique_vals = list(set(raw_values))
-                for row_idx, val in enumerate(raw_values):
-                    result[row_idx, col_idx] = unique_vals.index(val)
-
-        mean = np.mean(result, axis=0)
-        std = np.std(result, axis=0)
-        std[std == 0] = 1.0
-        return (result - mean) / std
+                filtered_vals = [v for v in raw_values if v is not None]
+                unique_vals = sorted(set(filtered_vals))
+                val_to_idx = {v: i for i, v in enumerate(unique_vals)}
+                encoded = np.zeros(len(raw_values), dtype=np.float64)
+                for row_idx, val in enumerate(raw_values):
+                    if val is not None and val in val_to_idx:
+                        encoded[row_idx] = float(val_to_idx[val])
+                    else:
+                        encoded[row_idx] = -1.0
+                if len(unique_vals) > 1:
+                    enc_min, enc_max = encoded[encoded >= 0].min(), encoded[encoded >= 0].max()
+                    if enc_max > enc_min:
+                        encoded[encoded >= 0] = (encoded[encoded >= 0] - enc_min) / (enc_max - enc_min)
+                result[:, col_idx] = encoded
+
+        return result
+
+    @staticmethod
+    def _is_nan(value: Any) -> bool:
+        if isinstance(value, float):
+            return np.isnan(value)
+        if isinstance(value, str):
+            return value.strip().lower() in ("nan", "na", "n/a", "null", "none", "")
+        return False
+
+    @staticmethod
+    def _interpolate_nans(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
+        if np.all(mask):
+            return np.zeros_like(arr)
+        valid_indices = np.where(~mask)[0]
+        nan_indices = np.where(mask)[0]
+        arr[nan_indices] = np.interp(nan_indices, valid_indices, arr[valid_indices])
+        return arr

     def _infer_column_type(self, values: List[Any]) -> str:
         numeric_count = 0
@@ -62,10 +95,14 @@
             if val is None:
                 continue
             try:
-                float(val)
-                numeric_count += 1
+                parsed = float(val)
+                if not np.isnan(parsed):
+                    numeric_count += 1
+                else:
+                    numeric_count += 1
             except (ValueError, TypeError):
                 non_numeric_count += 1
+
         total = numeric_count + non_numeric_count
         if total == 0:
             return "numerical"
@@ -82,36 +119,58 @@
         self, data: List[Dict[str, Any]], reduce_fn=None
     ) -> Dict[str, float]:
         if reduce_fn is None:
-            reduce_fn = np.mean
-
+            reduce_fn = np.nanmean
+
         result = {}
-        for key in data[0]:
-            values = [row[key] for row in data]
-            try:
-                numeric_values = [float(v) for v in values if v is not None]
-                result[key] = float(reduce_fn(numeric_values))
-            except (ValueError, TypeError):
-                result[key] = 0.0
+        if not data:
+            return result
+
+        all_keys = set()
+        for row in data:
+            all_keys.update(row.keys())
+
+        for key in sorted(all_keys):
+            values = [row.get(key) for row in data]
+            numeric_values = []
+            for v in values:
+                if v is None:
+                    numeric_values.append(np.nan)
+                    continue
+                try:
+                    parsed = float(v)
+                    numeric_values.append(parsed)
+                except (ValueError, TypeError):
+                    continue
+
+            if numeric_values:
+                arr = np.array(numeric_values, dtype=np.float64)
+                if np.all(np.isnan(arr)):
+                    result[key] = 0.0
+                else:
+                    result[key] = float(reduce_fn(arr))
+            else:
+                result[key] = 0.0
+
         return result
diff --git a/src/executor.py b/src/executor.py
--- a/src/executor.py
+++ b/src/executor.py
@@ -8,52 +8,89 @@
 import threading
 import queue
-from typing import List, Callable, Any, Dict, Optional
+from typing import List, Callable, Any, Dict, Optional, Tuple
 from concurrent.futures import ThreadPoolExecutor, as_completed
 from dataclasses import dataclass, field
+import contextlib
+import time


 @dataclass
 class WorkerState:
     partial_result: Any = None
-    error: Optional[str] = None
+    error: Optional[Exception] = None
     is_complete: bool = False
+    start_time: float = 0.0
+    end_time: float = 0.0
+    retry_count: int = 0


 class ParallelExecutor:
-    def __init__(self, num_workers: int = 4, timeout: float = 30.0):
+    def __init__(self, num_workers: int = 4, timeout: float = 30.0, max_retries: int = 2):
         self.num_workers = num_workers
         self.timeout = timeout
-        self._lock = threading.Lock()
+        self.max_retries = max_retries
+        self._lock = threading.RLock()
+        self._result_lock = threading.Lock()
         self._shared_state: Dict[str, Any] = {}
         self._worker_states: Dict[int, WorkerState] = {}
+        self._barrier = threading.Barrier(num_workers, timeout=timeout)
+        self._error_event = threading.Event()

     def execute_parallel(
         self, tasks: List[Callable], merge_fn: Optional[Callable] = None
     ) -> Any:
-        results = []
-        shared_accumulator = {"total": 0.0, "count": 0}
-
+        results: Dict[int, Any] = {}
+        shared_accumulator = {"total": 0.0, "count": 0, "errors": []}
+        self._error_event.clear()
+
+        for i in range(len(tasks)):
+            self._worker_states[i] = WorkerState(start_time=time.monotonic())
+
         def worker_wrapper(task_idx: int, task_fn: Callable) -> Any:
-            state = WorkerState()
-            self._worker_states[task_idx] = state
+            state = self._worker_states[task_idx]
+            attempts = 0
+            last_error = None
+
+            while attempts <= self.max_retries:
+                if self._error_event.is_set():
+                    state.error = RuntimeError("Cancelled due to sibling failure")
+                    return None
+
+                try:
+                    result = task_fn()
+                    state.partial_result = result
+
+                    with self._result_lock:
+                        if isinstance(result, (int, float)):
+                            shared_accumulator["total"] += result
+                            shared_accumulator["count"] += 1
+                        elif isinstance(result, dict):
+                            for k, v in result.items():
+                                if isinstance(v, (int, float)):
+                                    prev = shared_accumulator.get(k, 0.0)
+                                    shared_accumulator[k] = prev + v
+
+                    state.is_complete = True
+                    state.end_time = time.monotonic()
+                    return result
+
+                except Exception as exc:
+                    last_error = exc
+                    attempts += 1
+                    state.retry_count = attempts
+                    if attempts > self.max_retries:
+                        state.error = last_error
+                        state.end_time = time.monotonic()
+                        with self._result_lock:
+                            shared_accumulator["errors"].append(
+                                {"task": task_idx, "error": str(last_error)}
+                            )
+                        return None
+                    time.sleep(0.01 * (2 ** attempts))
+
+            return None
+
+        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
+            futures = {
+                pool.submit(worker_wrapper, idx, fn): idx
+                for idx, fn in enumerate(tasks)
+            }
+            for future in as_completed(futures, timeout=self.timeout):
+                idx = futures[future]
+                try:
+                    res = future.result(timeout=self.timeout)
+                    if res is not None:
+                        results[idx] = res
+                except Exception as exc:
+                    self._worker_states[idx].error = exc

-            try:
-                result = task_fn()
-                state.partial_result = result
-
-                shared_accumulator["total"] += result if isinstance(result, (int, float)) else 0
-                shared_accumulator["count"] += 1
-
-                state.is_complete = True
-                return result
-            except Exception as e:
-                state.error = str(e)
-                return None
-
-        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
-            futures = {
-                executor.submit(worker_wrapper, idx, fn): idx
-                for idx, fn in enumerate(tasks)
-            }
-            for future in as_completed(futures, timeout=self.timeout):
-                result = future.result()
-                if result is not None:
-                    results.append(result)
-
         if merge_fn:
-            return merge_fn(results)
-        return results
+            ordered = [results[i] for i in sorted(results.keys())]
+            return merge_fn(ordered)
+        return [results[i] for i in sorted(results.keys())]
diff --git a/src/connection_manager.py b/src/connection_manager.py
--- a/src/connection_manager.py
+++ b/src/connection_manager.py
@@ -5,55 +5,93 @@
 import threading
 import time
-from typing import Optional, Dict
+from typing import Optional, Dict, Set
 from dataclasses import dataclass
+from contextlib import contextmanager
+import logging
+
+logger = logging.getLogger(__name__)


 @dataclass
 class Connection:
     conn_id: int
     created_at: float
     last_used: float
     is_checked_out: bool = False
+    checked_out_at: float = 0.0
+    idle_timeout: float = 300.0


 class ConnectionPool:
-    def __init__(self, max_size: int = 10, idle_timeout: float = 300.0):
+    def __init__(
+        self, max_size: int = 10, idle_timeout: float = 300.0, checkout_timeout: float = 600.0
+    ):
         self.max_size = max_size
         self.idle_timeout = idle_timeout
+        self.checkout_timeout = checkout_timeout
         self._pool: Dict[int, Connection] = {}
+        self._checked_out: Set[int] = set()
         self._lock = threading.Lock()
         self._next_id = 0
+        self._total_created = 0
+        self._total_destroyed = 0
+        self._cleanup_running = False

     def acquire(self) -> Connection:
         with self._lock:
             for conn_id, conn in self._pool.items():
                 if not conn.is_checked_out:
                     conn.is_checked_out = True
                     conn.last_used = time.time()
+                    conn.checked_out_at = time.time()
+                    self._checked_out.add(conn_id)
                     return conn
-
+
             if len(self._pool) < self.max_size:
                 conn = self._create_connection()
                 conn.is_checked_out = True
+                conn.checked_out_at = time.time()
                 self._pool[conn.conn_id] = conn
+                self._checked_out.add(conn.conn_id)
                 return conn
-
-        raise RuntimeError("Connection pool exhausted")
+
+            stale = self._find_stale_checkout()
+            if stale is not None:
+                logger.warning(f"Reclaiming stale checked-out connection {stale.conn_id}")
+                stale.last_used = time.time()
+                stale.checked_out_at = time.time()
+                return stale
+
+        raise RuntimeError(
+            f"Connection pool exhausted (max_size={self.max_size}, "
+            f"checked_out={len(self._checked_out)})"
+        )

     def release(self, conn: Connection) -> None:
         with self._lock:
             if conn.conn_id in self._pool:
                 conn.is_checked_out = False
                 conn.last_used = time.time()
+                conn.checked_out_at = 0.0
+                self._checked_out.discard(conn.conn_id)

     def cleanup(self) -> int:
-        now = time.time()
-        to_remove = []
         with self._lock:
-            for conn_id, conn in self._pool.items():
-                if not conn.is_checked_out and (now - conn.last_used) > self.idle_timeout:
-                    to_remove.append(conn_id)
-
-            for conn_id in to_remove:
-                del self._pool[conn_id]
-
+            if self._cleanup_running:
+                return 0
+            self._cleanup_running = True
+
+        now = time.time()
+        to_remove = []
+        try:
+            with self._lock:
+                for conn_id, conn in self._pool.items():
+                    if not conn.is_checked_out and (now - conn.last_used) > self.idle_timeout:
+                        to_remove.append(conn_id)
+                    elif conn.is_checked_out and conn.checked_out_at > 0:
+                        if (now - conn.checked_out_at) > self.checkout_timeout:
+                            logger.warning(
+                                f"Connection {conn_id} checked out for "
+                                f"{now - conn.checked_out_at:.1f}s, forcing release"
+                            )
+                            conn.is_checked_out = False
+                            conn.checked_out_at = 0.0
+                            conn.last_used = now
+                            self._checked_out.discard(conn_id)
+
+                for conn_id in to_remove:
+                    del self._pool[conn_id]
+                    self._checked_out.discard(conn_id)
+                    self._total_destroyed += 1
+        finally:
+            with self._lock:
+                self._cleanup_running = False
+
         return len(to_remove)
"""

# ---------------------------------------------------------------------------
# Main loop — identical request pattern to model_client.compute_score_batch()
# ---------------------------------------------------------------------------
def build_payload(num_batches: int, batch_size: int) -> dict:
    batches = [
        {
            "batch_id": f"keepalive_{i}",
            "data": {
                "issue": MOCK_ISSUE,
                "patch_list": [MOCK_PATCH] * batch_size,
            },
        }
        for i in range(num_batches)
    ]
    return {"batches": batches}


def send_keepalive(payload: dict, server_url: str, max_retries: int = 3) -> bool:
    for attempt in range(max_retries):
        try:
            response = requests.post(
                server_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                proxies={"http": None, "https": None},  # no proxy, same as model_client
            )
            response.raise_for_status()
            return True
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            if attempt < max_retries - 1:
                print(f"  Request failed: {e}. Retry {attempt + 1}/{max_retries - 1}...")
            else:
                print(f"  Request failed after {max_retries} attempts: {e}")
    return False


def main():
    payload = build_payload(NUM_BATCHES, BATCH_SIZE)
    total_items = NUM_BATCHES * BATCH_SIZE
    payload_bytes = len(__import__("json").dumps(payload).encode())

    print("=== RM Keep-Alive Started ===")
    print(f"  Server:   {SERVER_URL}")
    print(f"  Interval: {INTERVAL}s")
    print(f"  Payload:  {NUM_BATCHES} batches × {BATCH_SIZE} patches = {total_items} items ({payload_bytes:,} bytes)")
    print()

    while True:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        ok = send_keepalive(payload, SERVER_URL)
        if ok:
            print(f"[{ts}] Keep-alive OK")
        else:
            print(f"[{ts}] Keep-alive FAILED")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
