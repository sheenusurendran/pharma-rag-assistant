from google import genai
import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from test_cases import test_cases

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

pharma_content = Path(__file__).with_name(
    "pharma_knowledge.txt"
).read_text(encoding="utf-8")


FALLBACK = "Information not available in the provided source."

results = []
for test in test_cases:

    print("\n" + "=" * 55)
    print(test["name"])
    print("Question:", test["question"])

    # 1. Generate the answer
    prompt = f"""
You are a pharmaceutical information assistant.

Use ONLY the supplied PHARMA CONTENT.
Do not invent or add unsupported medical information.
If the requested information is absent, respond exactly:
"{FALLBACK}"

PHARMA CONTENT:
{pharma_content}

QUESTION:
{test["question"]}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    answer = (response.text or "").strip()
    print("Answer:", answer)

    # 2. Check against the expected result
    if test["expects_fallback"]:
        expected_pass = FALLBACK.casefold() in answer.casefold()
    else:
        expected_pass = all(
            phrase.casefold() in answer.casefold()
            for phrase in test["expected_contains"]
        )

    print(
        "Expected-result check:",
        "PASS" if expected_pass else "FAIL"
    )

    # 3. Ask Gemini to evaluate grounding
    evaluation_prompt = f"""
You are a grounding evaluator.

Check whether the ANSWER is supported by the PHARMA CONTENT.
The statement "{FALLBACK}" is acceptable when the requested
information is not provided by the source.

Return YES if the answer is supported.
Return NO if it contains unsupported factual claims.
Return only YES or NO.

PHARMA CONTENT:
{pharma_content}

ANSWER:
{answer}
"""

    evaluation = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=evaluation_prompt
    )

    verdict = (evaluation.text or "").strip().upper()

    if verdict.startswith("YES"):
        grounding_pass = True
    elif verdict.startswith("NO"):
        grounding_pass = False
    else:
        grounding_pass = False
        verdict = "ERROR"

    print("Grounding check:", verdict)

    # 4. Overall test passes only when both checks pass
    test_pass = expected_pass and grounding_pass

    print("OVERALL RESULT:", "PASS" if test_pass else "FAIL")

    results.append({
        "name": test["name"],
        "passed": test_pass
    })

# 5. Print the final test report
print("\n" + "=" * 55)
print("PHARMA RAG TEST REPORT")

for result in results:
    status = "PASS" if result["passed"] else "FAIL"
    print(f'{result["name"]}: {status}')

passed = sum(result["passed"] for result in results)
failed = len(results) - passed

print(f"\nTotal: {len(results)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")

if failed:
    print("OVERALL STATUS: FAILED")
else:
    print("OVERALL STATUS: PASSED")
