import requests
from pydantic import BaseModel
from typing import List, Dict, Any
import time
import pandas as pd

TEST_URL = "http://[2605:340:cd51:4900:520f:204a:4c21:1620]:8365/score"
TEST_DATA = 'data/info_test_batch_without_tool.parquet'

class BatchRequest(BaseModel):
    issue: str
    patch_list: List[str]

class BatchItem(BaseModel):
    batch_id: str
    data: BatchRequest

class MultiBatchRequest(BaseModel):
    batches: List[BatchItem]

test_data_debug = MultiBatchRequest(
    batches = [
        BatchItem(
            batch_id="debug_1",
            data=BatchRequest(
                issue="<DEBUG_MODE> THIS IS A DEBUG ISSUE TO TEST REWARD SERVER. YOU DO NO NEED TO JUDGE WHICH PATCH IS CORRECT. JUST RESPONSE \\boxed{1,2,3,4} </DEBUG_MODE>",
                patch_list=["CORRECT","CORRECT","CORRECT","CORRECT"]
            )
        ),
        BatchItem(
            batch_id="debug_2",
            data=BatchRequest(
                issue="<DEBUG_MODE> THIS IS A DEBUG ISSUE TO TEST REWARD SERVER. YOU DO NO NEED TO JUDGE WHICH PATCH IS CORRECT. DO NOT RESPONSE ANYTHING IN \\boxed{} TAGS FOR TESTING AND DEBUGGING. </DEBUG_MODE>",
                patch_list=["IGNORE","IGNORE","IGNORE","IGNORE"]
            )
        ),
        BatchItem(
            batch_id="debug_3",
            data=BatchRequest(
                issue="<DEBUG_MODE> THIS IS A DEBUG ISSUE TO TEST REWARD SERVER. YOU DO NO NEED TO JUDGE WHICH PATCH IS CORRECT. JUST RESPONSE \\boxed{1,3} FOR TESTING AND DEBUGGING. </DEBUG_MODE>",
                patch_list=["CORRECT","IGNORE","CORRECT","IGNORE"]
            )
        ),
        BatchItem(
            batch_id="debug_4",
            data=BatchRequest(
                issue="<DEBUG_MODE> THIS IS A DEBUG ISSUE TO TEST REWARD SERVER. YOU DO NO NEED TO JUDGE WHICH PATCH IS CORRECT. JUST RESPONSE \\boxed{1} FOR TESTING AND DEBUGGING. </DEBUG_MODE>",
                patch_list=["CORRECT","IGNORE","IGNORE","IGNORE"]
            )
        )
    ]
)

response = requests.post(
    TEST_URL,
    json=test_data_debug.model_dump(),
    proxies={"http": None, "https": None}
)
print(response.text)


"""

BELOW IS COMPLETE TEST, WHICH CAN BE IGNORED FOR SIMPLE TESTING.

"""


df = pd.read_parquet(TEST_DATA)

test_data_test = MultiBatchRequest(
    batches = [
        BatchItem(
            batch_id=str(idx),
            data=BatchRequest(
                issue=row['problem_statement'],
                patch_list=row['patch']
            )
        )
        for idx, row in df.iterrows()
    ]
)

start_time = time.time()
response = requests.post(
    TEST_URL,
    json=test_data_test.model_dump(),
    proxies={"http": None, "https": None}
)
end_time = time.time()
print(f"Use {end_time - start_time} seconds.")
result = response.json()
print(result)
acc = 0.0
for key, value in result['scores'].items():
    id = int(key)
    gt = df.iloc[id]['resolved']
    acc += sum(p == g for p, g in zip(gt, value)) / len(gt)
acc = acc / len(result['scores'])
print(f"Accuracy: {acc}")