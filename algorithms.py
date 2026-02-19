"""
经典算法集 - 基于 TheAlgorithms/Python
包含: 排序、搜索、字符串、链表、树、图、动态规划等
"""

# ==================== 排序算法 ====================

def bubble_sort(arr):
    """冒泡排序 O(n²)"""
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def selection_sort(arr):
    """选择排序 O(n²)"""
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def insertion_sort(arr):
    """插入排序 O(n²)"""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def merge_sort(arr):
    """归并排序 O(n log n)"""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    """归并辅助函数"""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def heap_sort(arr):
    """堆排序 O(n log n)"""
    def heapify(arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[l] > arr[largest]:
            largest = l
        if r < n and arr[r] > arr[largest]:
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)
    
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
    return arr

# ==================== 搜索算法 ====================

def binary_search(arr, target):
    """二分查找 O(log n)"""
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

def linear_search(arr, target):
    """线性查找 O(n)"""
    for i, v in enumerate(arr):
        if v == target:
            return i
    return -1

def jump_search(arr, target):
    """跳跃搜索 O(√n)"""
    import math
    n = len(arr)
    step = int(math.sqrt(n))
    prev = 0
    while arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    while arr[prev] < target:
        prev += 1
        if prev == min(step, n):
            return -1
    if arr[prev] == target:
        return prev
    return -1

# ==================== 字符串算法 ====================

def reverse_string(s):
    """反转字符串"""
    return s[::-1]

def is_palindrome(s):
    """回文判断"""
    s = s.lower().replace(" ", "")
    return s == s[::-1]

def longest_common_substring(s1, s2):
    """最长公共子串"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(2)]
    max_len = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i % 2][j] = dp[(i-1) % 2][j-1] + 1
                if dp[i % 2][j] > max_len:
                    max_len = dp[i % 2][j]
            else:
                dp[i % 2][j] = 0
    return max_len

def kmp_pattern_match(text, pattern):
    """KMP 模式匹配"""
    def compute_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps
    
    lps = compute_lps(pattern)
    i = j = 0
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == len(pattern):
            return i - j
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return -1

def rabin_karp(text, pattern):
    """Rabin-Karp 字符串匹配"""
    d = 256
    q = 101
    n, m = len(text), len(pattern)
    h = pow(d, m-1) % q
    p = t = 0
    for i in range(m):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q
    for i in range(n - m + 1):
        if p == t:
            if text[i:i+m] == pattern:
                return i
        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i+m])) % q
            if t < 0:
                t += q
    return -1

# ==================== 链表 ====================

class ListNode:
    """链表节点"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(head):
    """反转链表"""
    prev = None
    current = head
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    return prev

def linked_list_to_list(head):
    """链表转列表"""
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result

# ==================== 树 ====================

class TreeNode:
    """二叉树节点"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder_traversal(root):
    """中序遍历"""
    result = []
    def helper(node):
        if node:
            helper(node.left)
            result.append(node.val)
            helper(node.right)
    helper(root)
    return result

def preorder_traversal(root):
    """前序遍历"""
    result = []
    def helper(node):
        if node:
            result.append(node.val)
            helper(node.left)
            helper(node.right)
    helper(root)
    return result

def postorder_traversal(root):
    """后序遍历"""
    result = []
    def helper(node):
        if node:
            helper(node.left)
            helper(node.right)
            result.append(node.val)
    helper(root)
    return result

def level_order_traversal(root):
    """层序遍历"""
    if not root:
        return []
    result, queue = [], [root]
    while queue:
        node = queue.pop(0)
        result.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result

def tree_height(root):
    """树的高度"""
    if not root:
        return 0
    return 1 + max(tree_height(root.left), tree_height(root.right))

# ==================== 图算法 ====================

def bfs(graph, start):
    """广度优先搜索"""
    visited = set([start])
    queue = [start]
    result = []
    while queue:
        vertex = queue.pop(0)
        result.append(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return result

def dfs(graph, start):
    """深度优先搜索"""
    visited = set()
    result = []
    def helper(vertex):
        visited.add(vertex)
        result.append(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                helper(neighbor)
    helper(start)
    return result

def dijkstra(graph, start):
    """Dijkstra 最短路径"""
    import heapq
    dist = {v: float('inf') for v in graph}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v in graph[u]:
            alt = dist[u] + graph[u][v]
            if alt < dist[v]:
                dist[v] = alt
                heapq.heappush(pq, (alt, v))
    return dist

# ==================== 动态规划 ====================

def fibonacci_dp(n):
    """斐波那契 - 动态规划"""
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

def coin_change(coins, amount):
    """零钱兑换 - 最少硬币数"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i-coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

def longest_increasing_subsequence(arr):
    """最长递增子序列"""
    if not arr:
        return 0
    dp = [1] * len(arr)
    for i in range(1, len(arr)):
        for j in range(i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def knapsack(values, weights, capacity):
    """0-1 背包问题"""
    n = len(values)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]

# ==================== 数学算法 ====================

def gcd(a, b):
    """最大公约数 - 欧几里得算法"""
    while b:
        a, b = b, a % b
    return abs(a)

def lcm(a, b):
    """最小公倍数"""
    return abs(a * b) // gcd(a, b)

def is_prime(n):
    """素数判断"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def sieve_of_eratosthenes(limit):
    """埃拉托斯特尼筛法 - 求素数"""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def fast_power(base, exp):
    """快速幂运算"""
    result = 1
    while exp > 0:
        if exp % 2 == 1:
            result *= base
        base *= base
        exp //= 2
    return result

# ==================== 位运算 ====================

def count_bits(n):
    """计算二进制中1的个数"""
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count

def reverse_bits(n, bits=32):
    """反转二进制位"""
    result = 0
    for i in range(bits):
        if n & (1 << i):
            result |= 1 << (bits - 1 - i)
    return result

def power_of_two(n):
    """判断是否为2的幂"""
    return n > 0 and (n & (n - 1)) == 0

# ==================== 加密算法 ====================

def caesar_cipher(text, shift, decode=False):
    """凯撒密码"""
    if decode:
        shift = -shift
    result = []
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            result.append(chr((ord(char) - base + shift) % 26 + base))
        else:
            result.append(char)
    return ''.join(result)

def xor_cipher(text, key):
    """XOR 加密"""
    return ''.join(chr(ord(c) ^ key for c in text)

# ==================== 其他算法 ====================

def two_sum(nums, target):
    """两数之和"""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

def three_sum(nums):
    """三数之和"""
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                left += 1
                right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
    return result

def max_subarray(nums):
    """最大子数组和 - Kadane算法"""
    max_sum = nums[0]
    current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

# ==================== 测试 ====================

def run_tests():
    print("🧪 运行算法测试...")
    
    # 排序测试
    arr = [64, 34, 25, 12, 22, 11, 90]
    assert bubble_sort(arr.copy()) == [11, 12, 22, 25, 34, 64, 90]
    assert merge_sort(arr.copy()) == [11, 12, 22, 25, 34, 64, 90]
    print("✅ 排序算法测试通过")
    
    # 搜索测试
    arr = [1, 3, 5, 7, 9, 11, 13]
    assert binary_search(arr, 7) == 3
    assert binary_search(arr, 6) == -1
    print("✅ 搜索算法测试通过")
    
    # 字符串测试
    assert is_palindrome("racecar") == True
    assert is_palindrome("hello") == False
    print("✅ 字符串算法测试通过")
    
    # 数学测试
    assert gcd(48, 18) == 6
    assert is_prime(17) == True
    assert is_prime(18) == False
    print("✅ 数学算法测试通过")
    
    # 位运算测试
    assert count_bits(7) == 3
    assert power_of_two(8) == True
    print("✅ 位运算测试通过")
    
    print("🎉 所有算法测试通过!")

if __name__ == "__main__":
    print("🐍 经典算法集 - TheAlgorithms Python")
    print("=" * 50)
    
    # 示例运行
    print("\n📊 排序算法:")
    arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"  原始: {arr}")
    print(f"  归并排序: {merge_sort(arr.copy())}")
    
    print("\n📊 搜索算法:")
    arr = [1, 3, 5, 7, 9, 11, 13]
    print(f"  二分查找 7: 位置 {binary_search(arr, 7)}")
    
    print("\n📊 字符串:")
    print(f"  'hello' 回文: {is_palindrome('hello')}")
    print(f"  'racecar' 回文: {is_palindrome('racecar')}")
    
    print("\n📊 数学:")
    print(f"  GCD(48, 18): {gcd(48, 18)}")
    print(f"  17 是素数: {is_prime(17)}")
    print(f"  100以内素数: {sieve_of_eratosthenes(20)}")
    
    print("\n📊 动态规划:")
    print(f"  斐波那契(10): {fibonacci_dp(10)}")
    print(f"  零钱兑换 [1,2,5], 11: {coin_change([1,2,5], 11)}")
    print(f"  最大子数组和: {max_subarray([-2,1,-3,4,-1,2,1,-5,4])}")
    
    print("\n📊 加密:")
    print(f"  凯撒密码 'hello' 偏移3: {caesar_cipher('hello', 3)}")
    print(f"  还原: {caesar_cipher('khoor', 3, decode=True)}")
    
    print("\n" + "=" * 50)
    run_tests()
