import torch
import torch.nn as nn

# setup.py를 통해 빌드된 C++ 확장 모듈 임포트
import fused_ln_gelu_cuda


class FusedLNGELUFunction(torch.autograd.Function):
    """
    Fused LayerNorm + GELU의 Autograd 연산 클래스
    """

    @staticmethod
    def forward(ctx, x, gamma, beta, epsilon=1e-5):
        """
        순전파 (Forward Pass)
        """
        # C++ CUDA 커널 호출 (반환값: 출력 y, 평균 mean, 역표준편차 rsigma)
        y, mean, rsigma = fused_ln_gelu_cuda.forward(x, gamma, beta, epsilon)

        # 역전파(Backward) 계산에 필요한 텐서들을 ctx에 저장
        ctx.save_for_backward(x, gamma, beta, mean, rsigma)

        return y

    @staticmethod
    def backward(ctx, dy):
        """
        역전파 (Backward Pass)
        """
        # 저장했던 텐서 복원
        x, gamma, beta, mean, rsigma = ctx.saved_tensors

        # dy가 contiguous하지 않을 경우를 대비
        if not dy.is_contiguous():
            dy = dy.contiguous()

        # C++ CUDA 역전파 커널 호출 (반환값: dx, dgamma, dbeta)
        dx, dgamma, dbeta = fused_ln_gelu_cuda.backward(
            dy, x, gamma, beta, mean, rsigma
        )

        # forward()의 인자 개수(x, gamma, beta, epsilon)와 일치하게 return
        # epsilon은 학습 대상 가중치가 아니므로 None 반환
        return dx, dgamma, dbeta, None


class FusedLNGELU(nn.Module):
    """
    nn.Module 형태의 사용자 친화적 래퍼 클래스
    """

    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # LayerNorm 가중치(gamma)와 편향(beta) 선언
        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        # Autograd Function의 apply() 메서드를 통해 실행
        return FusedLNGELUFunction.apply(x, self.gamma, self.beta, self.eps)