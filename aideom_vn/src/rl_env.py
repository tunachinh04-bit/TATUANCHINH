import numpy as np
from typing import Tuple, Dict, Any, Optional

try:
    import gymnasium as gym
    from gymnasium import spaces
    PARENT_CLASS = gym.Env
except ImportError:
    gym = None
    class Discrete:
        def __init__(self, n):
            self.n = n
        def sample(self):
            return int(np.random.randint(self.n))
    spaces = type("spaces", (), {"Discrete": Discrete})
    PARENT_CLASS = object

class VietnamEconomyEnv(PARENT_CLASS):
    """
    Môi trường Gymnasium mô phỏng kinh tế Việt Nam trong kỷ nguyên AI (Bài 11).
    Không gian trạng thái gồm 81 trạng thái rời rạc đại diện cho sự kết hợp của:
    1. Trình độ sẵn sàng AI (AI Readiness): 0 (Thấp), 1 (Trung bình), 2 (Cao) -> 3 trạng thái
    2. Tốc độ tăng trưởng GDP (GDP Growth): 0 (Chậm), 1 (Trung bình), 2 (Nhanh) -> 3 trạng thái
    3. Rủi ro thất nghiệp công nghệ (Labor Risk): 0 (Thấp), 1 (Trung bình), 2 (Cao) -> 3 trạng thái
    4. Tỷ lệ lao động qua đào tạo (Trained Labor): 0 (Thấp), 1 (Trung bình), 2 (Cao) -> 3 trạng thái
    Tổng số trạng thái = 3 * 3 * 3 * 3 = 81.
    
    Không gian hành động gồm 5 chính sách vĩ mô (Actions):
    - Action 0: Tập trung đầu tư hạ tầng vật chất (Infra Focus)
    - Action 1: Đẩy mạnh phát triển AI & Công nghệ số (AI & Digital Focus)
    - Action 2: Ưu tiên đào tạo nhân lực số (Human Capital Focus)
    - Action 3: Phát triển cân bằng (Balanced Investment)
    - Action 4: Tăng cường an sinh xã hội & đào tạo lại (Social Safety Net & Retraining)
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self):
        if PARENT_CLASS != object:
            super(VietnamEconomyEnv, self).__init__()
        
        # 5 hành động rời rạc
        self.action_space = spaces.Discrete(5)
        
        # 81 trạng thái rời rạc
        self.observation_space = spaces.Discrete(81)
        
        # Khởi tạo trạng thái ban đầu: Trình độ trung bình (AI=1, GDP=1, Risk=1, Train=1)
        # Cách giải mã trạng thái state -> (ai, gdp, risk, train):
        # state = ai * 27 + gdp * 9 + risk * 3 + train
        self.state = 1 * 27 + 1 * 9 + 1 * 3 + 1
        self.steps = 0
        self.max_steps = 20  # Một đợt hoạch định chính sách kéo dài 20 năm (hoặc 20 bước)
        
    def _decode_state(self, state: int) -> Tuple[int, int, int, int]:
        ai = state // 27
        rem = state % 27
        gdp = rem // 9
        rem = rem % 9
        risk = rem // 3
        train = rem % 3
        return ai, gdp, risk, train
        
    def _encode_state(self, ai: int, gdp: int, risk: int, train: int) -> int:
        # Giới hạn các giá trị trong khoảng [0, 2]
        ai = int(np.clip(ai, 0, 2))
        gdp = int(np.clip(gdp, 0, 2))
        risk = int(np.clip(risk, 0, 2))
        train = int(np.clip(train, 0, 2))
        return ai * 27 + gdp * 9 + risk * 3 + train
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        if PARENT_CLASS != object:
            super().reset(seed=seed)
        self.steps = 0
        # Bắt đầu tại trạng thái ổn định cơ sở (AI=1, GDP=1, Risk=1, Trained=1) -> state = 40
        self.state = self._encode_state(1, 1, 1, 1)
        return self.state, {}
        
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        self.steps += 1
        ai, gdp, risk, train = self._decode_state(self.state)
        
        # Lưu các giá trị cũ để phân tích
        prev_ai, prev_gdp, prev_risk, prev_train = ai, gdp, risk, train
        
        # Quy tắc chuyển đổi trạng thái mang tính kinh tế vĩ mô
        if action == 0:  # Tập trung hạ tầng
            gdp = gdp + 1 if np.random.rand() < 0.7 else gdp
            # Đầu tư hạ tầng truyền thống không giúp tăng trình độ AI hay nhân lực nhiều
            ai = ai - 1 if np.random.rand() < 0.2 else ai
            
        elif action == 1:  # Tập trung công nghệ AI & số hóa
            ai = ai + 1 if np.random.rand() < 0.8 else ai
            gdp = gdp + 1 if np.random.rand() < 0.5 else gdp
            # AI phát triển nhanh làm tăng rủi ro thất nghiệp nếu không đào tạo kịp thời
            if train < 2:
                risk = risk + 1 if np.random.rand() < 0.8 else risk
            else:
                risk = risk + 1 if np.random.rand() < 0.3 else risk
                
        elif action == 2:  # Ưu tiên nhân lực đào tạo
            train = train + 1 if np.random.rand() < 0.8 else train
            # Nhân lực chất lượng cao giúp giảm rủi ro thất nghiệp và tăng khả năng sẵn sàng AI
            risk = risk - 1 if np.random.rand() < 0.6 else risk
            ai = ai + 1 if np.random.rand() < 0.4 else ai
            
        elif action == 3:  # Phát triển cân bằng
            # Tăng nhẹ ở nhiều khía cạnh
            if np.random.rand() < 0.5:
                gdp = min(2, gdp + 1)
            if np.random.rand() < 0.4:
                ai = min(2, ai + 1)
            if np.random.rand() < 0.4:
                train = min(2, train + 1)
            # Rủi ro thất nghiệp được kiềm chế
            risk = risk - 1 if np.random.rand() < 0.3 else risk
            
        elif action == 4:  # An sinh xã hội & Đào tạo lại
            # Giảm rủi ro thất nghiệp mạnh mẽ và tăng nhẹ trình độ nhân lực
            risk = risk - 1 if np.random.rand() < 0.9 else risk
            train = train + 1 if np.random.rand() < 0.5 else train
            # Giảm nhẹ tài lực đầu tư phát triển gdp ngắn hạn
            if np.random.rand() < 0.1:
                gdp = max(0, gdp - 1)
                
        # Giới hạn trạng thái trước khi tính reward
        ai = int(np.clip(ai, 0, 2))
        gdp = int(np.clip(gdp, 0, 2))
        risk = int(np.clip(risk, 0, 2))
        train = int(np.clip(train, 0, 2))

        # Cập nhật trạng thái mới
        self.state = self._encode_state(ai, gdp, risk, train)
        
        # --- HÀM THƯỞNG PHÚC LỢI KINH TẾ (REWARD FUNCTION) ---
        # GDP tăng trưởng nhanh (gdp=2) đem lại điểm cộng lớn
        gdp_reward = [0.0, 10.0, 25.0][gdp]
        
        # Mức độ sẵn sàng AI đem lại hiệu suất kinh tế dài hạn
        ai_reward = [0.0, 5.0, 15.0][ai]
        
        # Nhân lực trình độ cao giúp gia tăng năng suất bền vững
        train_reward = [0.0, 5.0, 12.0][train]
        
        # Thất nghiệp cao (risk=2) chịu hình phạt rất nặng (phúc lợi xã hội giảm sâu)
        risk_penalty = [0.0, -8.0, -30.0][risk]
        
        # Chi phí của các hành động chính sách (cost of intervention)
        action_cost = {0: -4.0, 1: -6.0, 2: -5.0, 3: -7.0, 4: -3.0}[action]
        
        # Phạt nếu để xảy ra khủng hoảng kép (GDP kém và Thất nghiệp cao)
        crisis_penalty = -20.0 if (gdp == 0 and risk == 2) else 0.0
        
        reward = gdp_reward + ai_reward + train_reward + risk_penalty + action_cost + crisis_penalty
        
        # Kiểm tra trạng thái kết thúc
        terminated = self.steps >= self.max_steps
        truncated = False
        
        info = {
            'ai_level': ai,
            'gdp_growth': gdp,
            'labor_risk': risk,
            'trained_labor': train,
            'step_index': self.steps
        }
        
        return self.state, float(reward), terminated, truncated, info

def train_q_learning(
    env: VietnamEconomyEnv,
    episodes: int = 2000,
    alpha: float = 0.1,      # Học suất (learning rate)
    gamma: float = 0.9,      # Hệ số chiết khấu (discount factor)
    epsilon: float = 1.0,    # Tham số khám phá ban đầu
    min_epsilon: float = 0.05,
    decay_rate: float = 0.995
) -> Tuple[np.ndarray, list]:
    """
    Huấn luyện Agent bằng thuật toán Q-learning rời rạc trên môi trường VietnamEconomyEnv.
    
    Returns:
        Tuple[np.ndarray, list]:
            - Q_table: Ma trận Q-value kích thước (81, 5).
            - rewards_history: Lịch sử tổng phần thưởng qua các tập huấn luyện.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Khởi tạo Q-table bằng 0
    Q_table = np.zeros((n_states, n_actions))
    rewards_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Chọn hành động bằng chiến lược Epsilon-Greedy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Khám phá (Exploration)
            else:
                action = np.argmax(Q_table[state, :])  # Khai thác (Exploitation)
                
            # Thực thi bước hành động
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Cập nhật giá trị Q bằng phương trình Bellman sửa đổi
            best_next_action = np.argmax(Q_table[next_state, :])
            td_target = reward + gamma * Q_table[next_state, best_next_action]
            td_error = td_target - Q_table[state, action]
            Q_table[state, action] += alpha * td_error
            
            state = next_state
            total_reward += reward
            
        # Giảm dần mức độ khám phá epsilon
        epsilon = max(min_epsilon, epsilon * decay_rate)
        rewards_history.append(total_reward)
        
    return Q_table, rewards_history
