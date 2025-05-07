class Node:
    def __init__(self, level, profit, weight, count, quantities):
        self.level = level
        self.profit = profit
        self.weight = weight
        self.count = count
        self.quantities = quantities  # List of quantities for each item



class KnapSackBranchNBound:
    @staticmethod
    def bound(node, n, W, items, k):
        if node.weight >= W or node.count >= k:
            return 0

        profit_bound = node.profit
        j = node.level + 1

        if j < n:
            additional_items = min(k - node.count, (W - node.weight) // items[j][1])
            profit_bound += items[j][0] * additional_items

        return profit_bound

    @classmethod
    def fit(cls, W, items, k):
        items = sorted(items, key=lambda x: x[0] / x[1], reverse=True)
        n = len(items)

        Q = []
        v = Node(-1, 0, 0, 0, [0] * n)
        Q.append(v)

        max_profit = 0
        best_quantities = None

        while Q:
            v = Q.pop(0)

            if v.level + 1 < n:
                next_level = v.level + 1

                for i in range(k - v.count + 1):
                    new_weight = v.weight + items[next_level][1] * i
                    new_count = v.count + i

                    if new_weight <= W and new_count <= k:
                        new_quantities = v.quantities.copy()
                        new_quantities[next_level] += i

                        u = Node(next_level, v.profit + items[next_level][0] * i, new_weight, new_count, new_quantities)

                        if u.profit > max_profit:
                            max_profit = u.profit
                            best_quantities = new_quantities

                        if cls.bound(u, n, W, items, k) > max_profit:
                            Q.append(u)

        return max_profit, best_quantities
