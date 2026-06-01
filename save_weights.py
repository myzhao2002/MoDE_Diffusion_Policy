import torch
from mode.training_libero import train  # 假设这是你的入口
# 你的训练逻辑里应该有一个 model 对象
# 最简单的方法是直接利用 trainer 实例
# 如果你无法直接获取 trainer，你可以利用 wandb 的恢复机制，
# 或者如果你的 model 类支持，直接用 torch.save
print("模型权重已在内存中，请确保你已经实例化了 model 并加载了最佳权重。")
