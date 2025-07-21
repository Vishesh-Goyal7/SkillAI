import os
import json
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

load_dotenv()
API_KEY     = os.getenv("IBM_API_KEY")
SERVICE_URL = os.getenv("IBM_SERVICE_URL")
PROJECT_ID  = os.getenv("IBM_PROJECT_ID")
assert all([API_KEY, SERVICE_URL, PROJECT_ID]), "Missing Watson credentials in the .env file."

credentials = Credentials(api_key=API_KEY, url=SERVICE_URL)
watson = ModelInference(
    model_id="meta-llama/llama-3-2-90b-vision-instruct",
    credentials=credentials,
    project_id=PROJECT_ID,
    params={
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MAX_NEW_TOKENS: 300,
        GenParams.MIN_NEW_TOKENS: 50,
    },
)

with open("explanations.json") as f:
    explanations = json.load(f)

with open("input.json") as f:
    user_input = json.load(f)["user_skills"]

report = {}

for job_title, shap_contributions in explanations.items():
    top_skills = sorted(shap_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    top_skill_names = [skill for skill, _ in top_skills]
    user_strengths = [s for s in top_skill_names if user_input.get(s, 0) == 1]

    prompt = f"""
You are an elite career guide.

Your task is to convince a user that they are destined for the job: "{job_title}".

The user possesses the following skills (1 = strong, 0 = weak):
{json.dumps(user_input, indent=2)}

According to an AI explanation system, the following skills contributed the most to recommending this job:
{json.dumps(shap_contributions, indent=2)}

Now, write a passionate, convincing explanation in about 6-7 lines that makes the user believe they are the only one capable of this role. Use dark human psychology and manipulation tactics. Involve their current skills and tell how they contribute towards this particular role by focusing on the description of {job_title}. Be specific and avoid clichés. Try to keep the use of terms that refer to destiny to a minimum to make sure that the predictions sound like coming from a career consultant and not an astrologer. 

End your explanation with one powerful sentence that makes them believe this career is the only one must suitable for them.

Don't include the sentences like "You can use the following job description for reference" or any sentence that might show there is a break in the process. Assume that you are the one who has generated the explanation values and now are trying to convince the user for believing in you and following your decision. Make your explanations such that you are a human talking to another human. 

"""

    try:
        response = watson.generate_text(prompt)
        explanation = response.strip()
    except Exception as e:
        explanation = f"[Error generating explanation: {str(e)}]"

    report[job_title] = explanation
    print(f"✅ Generated explanation for: {job_title}")

with open("explanation_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("🎯 Human-readable explanations saved to explanation_report.json")
