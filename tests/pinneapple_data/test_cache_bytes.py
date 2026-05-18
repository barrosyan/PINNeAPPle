import torch
from pinneapple_data.cache_bytes import ByteLRUCache

def test_byte_lru_cache_eviction():
    c = ByteLRUCache(max_bytes=1024)
    c.put("a", torch.zeros(300, dtype=torch.uint8))  # 300B
    c.put("b", torch.zeros(800, dtype=torch.uint8))  # 800B => total 1100B -> evict "a"
    assert c.get("a") is None
    assert c.get("b") is not None
