import numpy as np

# 1. 파일 로드 (같은 폴더에 있다고 가정)
data = np.load('D_lunarlander_mixed_1m.npz')

# 2. 키(Key) 확인 - 가장 중요한 단계!
# 이 파일 안에 어떤 이름표로 데이터가 저장되어 있는지 목록을 보여줍니다.
print("Keys inside the file:", data.files) 
# 예상 출력: ['states', 'actions', 'rewards', ...] 또는 ['obs', 'acs', 'rews', ...]

# 3. 실제 데이터 꺼내기
# 위에서 확인한 키 이름을 그대로 사용해야 합니다.
states = data['states']  # (만약 키 이름이 'states'라면)
actions = data['actions']

print(f"States shape: {states.shape}")
print(f"Actions shape: {actions.shape}")