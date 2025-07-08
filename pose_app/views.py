import os
import numpy as np
import openai
from dotenv import load_dotenv
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import ReferencePose
from .utils import load_reference_pose, normalize_pose

# ✅ 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ 관절 세트 정의 (중심점, 점1, 점2)
angle_joints = [
    ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_WRIST"),
    ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_HIP"),
    ("LEFT_KNEE", "LEFT_HIP", "LEFT_ANKLE"),
    ("RIGHT_KNEE", "RIGHT_HIP", "RIGHT_ANKLE"),
    ("LEFT_HIP", "LEFT_KNEE", "LEFT_SHOULDER"),
    ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_SHOULDER"),
    ("NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER")
]

# ✅ 관절 이름 한글 맵핑
joint_name_map = {
    "LEFT_ELBOW": "왼쪽 팔꿈치", "RIGHT_ELBOW": "오른쪽 팔꿈치",
    "LEFT_SHOULDER": "왼쪽 어깨", "RIGHT_SHOULDER": "오른쪽 어깨",
    "LEFT_KNEE": "왼쪽 무릎", "RIGHT_KNEE": "오른쪽 무릎",
    "LEFT_HIP": "왼쪽 엉덩이", "RIGHT_HIP": "오른쪽 엉덩이",
    "NOSE": "목"
}

# ✅ 각도 계산 함수
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# ✅ 관절 각도 차이 계산 함수
def get_joint_angle_differences(ref_pose, user_pose):
    angle_diffs = {}
    for center, a, b in angle_joints:
        if all(j in ref_pose and j in user_pose for j in [a, center, b]):
            ref_angle = calculate_angle(ref_pose[a], ref_pose[center], ref_pose[b])
            user_angle = calculate_angle(user_pose[a], user_pose[center], user_pose[b])
            angle_diffs[center] = round(abs(ref_angle - user_angle), 2)
        else:
            angle_diffs[center] = None
    return angle_diffs

# ✅ GPT 프롬프트 생성 함수
def build_prompt(joint_names, angle_differences, threshold=10):
    feedback_parts = []
    for joint, diff in zip(joint_names, angle_differences):
        if diff is not None and diff > threshold:
            friendly_name = joint_name_map.get(joint, joint)
            feedback_parts.append(f"- {friendly_name}가 기준보다 {diff:.1f}도 차이가 납니다.")

    # threshold를 넘은 관절이 하나도 없으면 칭찬 프롬프트
    if not feedback_parts:
        return """당신은 피트니스 앱의 AI 코치입니다. 사용자의 현재 자세는 기준 자세와 거의 일치합니다.

사용자에게 딱 한 문장의 짧고 따뜻한 칭찬 멘트를 생성해주세요.

가이드라인:
- 딱 한 문장만 작성하세요.
- 형식적이지 않게, 부드러운 말투로.
- 현재 자세가 잘 유지되고 있다는 느낌을 주세요.

예시: “몸의 균형이 아주 잘 잡혀 있어요, 그대로만 유지해보세요!”
"""

    # 차이 큰 관절만 포함해서 피드백 요청
    detail = "\n".join(feedback_parts)
    return f"""

당신은 피트니스 앱의 AI 코치입니다. 사용자는 일반인 또는 시각적 제약이 있는 사람이며, 자세 인식 결과를 바탕으로 피드백을 받습니다.

아래는 사용자의 관절별 기준 자세와 실제 자세 간의 각도 차이 중, 차이가 큰 부위만 정리한 내용입니다.
해당 부위를 중심으로 다음 조건에 따라 피드백을 작성해주세요:

[관절별 각도 차이]
{detail}

작성 가이드:
- 문장은 딱딱하지 않게, 부드럽게 말하듯 작성하세요.
- 1문장은 문제 설명, 1문장은 교정 조언으로 구성하세요.
- 전문 용어는 피하고, 감각 중심 표현을 사용하세요.
- 칭찬은 생략하고, 교정 중심으로 작성하세요.
- 총 2문장 이내로 작성해주세요.

예시: "왼쪽 엉덩이가 오른쪽보다 높게 들려 있어요. 양 발에 고르게 체중을 실어서 중심을 가운데로 맞춰볼게요."

이제 사용자에게 피드백을 작성해 주세요.

"""


# ✅ GPT 호출 함수
def query_gpt(prompt, model="gpt-4o"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

# ✅ 각도 차이만 반환하는 API
class ComparePoseView(APIView):
    def post(self, request):
        pose_type = request.data.get('pose_type')
        user_pose = request.data.get('user_pose')

        if not pose_type or not user_pose:
            return Response({"error": "pose_type and user_pose are required."}, status=400)

        try:
            ref = ReferencePose.objects.filter(pose_type=pose_type).first()
            if not ref:
                return Response({"error": "ReferencePose not found."}, status=404)

            ref_pose = load_reference_pose(ref.csv_path, ref.row_index, pose_type)
            ref_pose_norm = normalize_pose(ref_pose)
            user_pose_norm = normalize_pose(user_pose)

            angle_diffs = get_joint_angle_differences(ref_pose_norm, user_pose_norm)
            return Response(angle_diffs)

        except Exception as e:
            return Response({"error": str(e)}, status=500)

# ✅ GPT 피드백까지 주는 API
class FeedbackView(APIView):
    def post(self, request):
        pose_type = request.data.get('pose_type')
        category = request.data.get('category')
        user_pose = request.data.get('user_pose')

        if not pose_type or not category or not user_pose:
            return Response({"error": "Missing pose_type, category, or user_pose"}, status=400)

        ref = ReferencePose.objects.filter(pose_type=pose_type, category=category).first()
        if not ref:
            return Response({"error": "ReferencePose not found."}, status=404)

        ref_pose = load_reference_pose(ref.csv_path, ref.row_index, pose_type)
        ref_pose_norm = normalize_pose(ref_pose)
        user_pose_norm = normalize_pose(user_pose)

        angle_diffs = get_joint_angle_differences(ref_pose_norm, user_pose_norm)
        joints = list(angle_diffs.keys())
        differences = list(angle_diffs.values())

        prompt = build_prompt(joints, differences, threshold=15)

        feedback = query_gpt(prompt)

        return Response({
            "feedback": feedback,
            "angle_differences": angle_diffs
        })

class CurrentPoseNameView(APIView):
    def get(self, request):
        pose_type = request.query_params.get("pose_type")
        category = request.query_params.get("category")

        if not pose_type or not category:
            return Response({"error": "pose_type and category required"}, status=400)

        ref = ReferencePose.objects.filter(pose_type=pose_type, category=category).first()
        if not ref:
            return Response({"error": "No matching pose found"}, status=404)

        return Response({"pose_name": ref.name})
