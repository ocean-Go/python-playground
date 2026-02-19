"""
Python Examples Collection
中等难度 Python 示例集
"""

# ==================== 1. 数据处理 ====================

def fibonacci(n):
    """斐波那契数列"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def fibonacci_iterative(n):
    """斐波那契数列 - 迭代版本"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n-1):
        a, b = b, a + b
    return b

def quicksort(arr):
    """快速排序"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def binary_search(arr, target):
    """二分查找"""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# ==================== 2. 文件处理 ====================

def count_lines(filepath):
    """统计文件行数"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    except FileNotFoundError:
        return -1

def word_frequency(text):
    """词频统计"""
    import re
    words = re.findall(r'\w+', text.lower())
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)

# ==================== 3. API 请求 ====================

import json

def fetch_json(url):
    """模拟 JSON API 请求"""
    # 这个函数在实际环境中可以用 requests 库
    mock_response = {
        "status": "success",
        "data": {
            "users": [
                {"id": 1, "name": "Alice", "age": 25},
                {"id": 2, "name": "Bob", "age": 30},
                {"id": 3, "name": "Charlie", "age": 28}
            ]
        }
    }
    return mock_response

def parse_api_response(response):
    """解析 API 响应"""
    if response.get("status") == "success":
        return response.get("data", {}).get("users", [])
    return []

# ==================== 4. 装饰器 ====================

def timer(func):
    """计时装饰器"""
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️ {func.__name__} 运行时间: {end-start:.4f}秒")
        return result
    return wrapper

def cache(func):
    """缓存装饰器"""
    memo = {}
    def wrapper(*args):
        if args not in memo:
            memo[args] = func(*args)
        return memo[args]
    return wrapper

@timer
def slow_function():
    import time
    time.sleep(0.5)
    return "完成!"

# ==================== 5. 类与继承 ====================

class Animal:
    """动物基类"""
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def speak(self):
        return "..."
    
    def info(self):
        return f"{self.name}, {self.age}岁"

class Dog(Animal):
    def __init__(self, name, age, breed="田园犬"):
        super().__init__(name, age)
        self.breed = breed
    
    def speak(self):
        return f"{self.name} 汪汪! 🐕"
    
    def fetch(self):
        return f"{self.name} 捡回球球!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} 喵喵! 🐱"
    
    def purr(self):
        return f"{self.name} 发出呼噜声..."

# ==================== 6. 上下文管理器 ====================

class FileManager:
    """文件管理器"""
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        return False

# ==================== 7. 生成器 ====================

def prime_generator(limit):
    """质数生成器"""
    for num in range(2, limit + 1):
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            yield num

def batch_generator(items, batch_size):
    """批量生成器"""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# ==================== 8. 测试函数 ====================

def run_all_tests():
    """运行所有测试"""
    print("🧪 开始测试...")
    
    # 测试斐波那契
    assert fibonacci(10) == 55
    assert fibonacci_iterative(10) == 55
    print("✅ 斐波那契测试通过")
    
    # 测试排序
    unsorted = [3, 1, 4, 1, 5, 9, 2, 6]
    sorted_arr = quicksort(unsorted)
    assert sorted_arr == [1, 1, 2, 3, 4, 5, 6, 9]
    print("✅ 快速排序测试通过")
    
    # 测试二分查找
    arr = [1, 3, 5, 7, 9, 11, 13]
    assert binary_search(arr, 7) == 3
    assert binary_search(arr, 6) == -1
    print("✅ 二分查找测试通过")
    
    # 测试类
    dog = Dog("旺财", 3)
    assert dog.speak() == "旺财 汪汪! 🐕"
    assert dog.fetch() == "旺财 捡回球球!"
    print("✅ 类继承测试通过")
    
    print("🎉 所有测试通过!")

# ==================== 主程序 ====================

if __name__ == "__main__":
    print("🐍 Python Examples Collection")
    print("=" * 40)
    
    # 1. 斐波那契
    print("\n📊 斐波那契数列 (迭代):")
    for i in range(10):
        print(f"  F({i}) = {fibonacci_iterative(i)}")
    
    # 2. 快速排序
    print("\n📈 快速排序:")
    arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"  原始: {arr}")
    print(f"  排序: {quicksort(arr)}")
    
    # 3. API 响应解析
    print("\n🌐 API 响应:")
    response = fetch_json("https://api.example.com")
    users = parse_api_response(response)
    for user in users:
        print(f"  - {user['name']}, {user['age']}岁")
    
    # 4. 动物类
    print("\n🐾 动物类:")
    dog = Dog("小白", 5, "萨摩耶")
    cat = Cat("咪咪", 2)
    print(f"  {dog.speak()}")
    print(f"  {dog.fetch()}")
    print(f"  {cat.speak()}")
    print(f"  {cat.purr()}")
    
    # 5. 质数生成器
    print("\n🔢 质数 (前10个):")
    primes = list(prime_generator(30))
    print(f"  {primes}")
    
    # 6. 运行测试
    print()
    run_all_tests()
