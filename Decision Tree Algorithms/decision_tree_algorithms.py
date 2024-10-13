import numpy as np
import pandas as pd
from math import log2
import graphviz

# Dữ liệu mẫu
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny',
                'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot',
                    'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)


# Hàm tính entropy
def entropy(y):
    # Tìm các lớp và số lượng của chúng trong y
    classes, counts = np.unique(y, return_counts=True)
    
    # Tính xác suất cho mỗi lớp
    probabilities = counts / len(y)
    
    # Tính entropy dựa trên xác suất
    return -sum(p * log2(p) for p in probabilities)


# Hàm tính information gain
def information_gain(X, y, feature):
    # Tính entropy tổng thể của tập dữ liệu
    total_entropy = entropy(y)
    
    # Lấy các giá trị duy nhất và số lượng của chúng trong đặc trưng
    values, counts = np.unique(X[feature], return_counts=True)
    
    # Tính entropy có trọng số cho từng giá trị của đặc trưng
    weighted_entropy = sum(counts[i] / len(y) * entropy(y[X[feature] == values[i]]) for i in range(len(values)))
    
    # Trả về thông tin thu được bằng cách trừ đi entropy có trọng số từ entropy tổng thể
    return total_entropy - weighted_entropy


# Hàm tính gain ratio (cho C4.5)
def gain_ratio(X, y, feature):
    # Tính thông tin thu được (information gain)
    ig = information_gain(X, y, feature)
    # Tính entropy của thuộc tính (split information)
    split_info = entropy(X[feature])
    # Trả về gain ratio, nếu split_info khác 0
    return ig / split_info if split_info != 0 else 0


# Hàm tính Gini index (cho CART)
def gini_index(y):
    # Tìm các lớp và số lượng của chúng trong y
    classes, counts = np.unique(y, return_counts=True)
    # Tính xác suất của mỗi lớp
    probabilities = counts / len(y)
    # Tính chỉ số Gini
    return 1 - sum(p ** 2 for p in probabilities)


# ID3 Algorithm
def id3(X, y, features):
    # Nếu tất cả các nhãn trong y đều giống nhau, trả về nhãn đó
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    
    # Nếu không còn đặc trưng nào để phân chia, trả về nhãn phổ biến nhất
    if len(features) == 0:
        return np.argmax(np.bincount(y))

    # Tìm đặc trưng tốt nhất để phân chia dựa trên information gain
    best_feature = max(features, key=lambda f: information_gain(X, y, f))
    tree = {best_feature: {}}

    # Duyệt qua từng giá trị của đặc trưng tốt nhất
    for value in np.unique(X[best_feature]):
        # Tạo các nút con X y cho mỗi giá trị duy nhất của đặc trưng đã chọn
        sub_X = X[X[best_feature] == value].drop(best_feature, axis=1)
        sub_y = y[X[best_feature] == value]
        # Loại bỏ đặc trưng tốt nhất khỏi danh sách đặc trưng
        sub_features = [f for f in features if f != best_feature]
        # Đệ quy xây dựng cây con cho giá trị hiện tại
        tree[best_feature][value] = id3(sub_X, sub_y, sub_features)

    # Trả về cây quyết định
    return tree

# Thuật toán C4.5
def c45(X, y, features):
    # Nếu tất cả các nhãn đều giống nhau, trả về nhãn đó
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    
    # Nếu không còn đặc trưng nào để phân chia, trả về nhãn phổ biến nhất
    if len(features) == 0:
        return np.argmax(np.bincount(y))

    # Chọn đặc trưng có gain ratio cao nhất làm nút gốc.
    best_feature = max(features, key=lambda f: gain_ratio(X, y, f))
    tree = {best_feature: {}}

    # Duyệt qua từng giá trị của đặc trưng tốt nhất
    for value in np.unique(X[best_feature]):
        # Tạo tập con dữ liệu và nhãn cho giá trị hiện tại
        sub_X = X[X[best_feature] == value].drop(best_feature, axis=1)
        sub_y = y[X[best_feature] == value]
        sub_features = [f for f in features if f != best_feature]
        
        # Đệ quy xây dựng cây con cho giá trị hiện tại
        tree[best_feature][value] = c45(sub_X, sub_y, sub_features)

    # Trả về cây quyết định
    return tree


# Thuật toán CART (Classification and Regression Trees)
def cart(X, y, features):
    # Nếu tất cả các nhãn trong y đều giống nhau, trả về nhãn đó
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    
    # Nếu không còn đặc trưng nào để phân chia, trả về nhãn phổ biến nhất
    if len(features) == 0:
        return np.argmax(np.bincount(y))

    # Tìm đặc trưng tốt nhất để phân chia dựa trên chỉ số Gini
    best_feature = min(features, key=lambda f: gini_index(y[X[f] == np.unique(X[f])[0]]) + gini_index(
        y[X[f] == np.unique(X[f])[1]]))
    
    # Khởi tạo cây với đặc trưng tốt nhất
    tree = {best_feature: {}}

    # Duyệt qua từng giá trị của đặc trưng tốt nhất
    for value in np.unique(X[best_feature]):
        # Tạo tập con X và y cho giá trị hiện tại
        sub_X = X[X[best_feature] == value].drop(best_feature, axis=1)
        sub_y = y[X[best_feature] == value]
        
        # Lấy danh sách đặc trưng còn lại sau khi loại bỏ đặc trưng tốt nhất
        sub_features = [f for f in features if f != best_feature]
        
        # Đệ quy xây dựng cây con cho giá trị hiện tại
        tree[best_feature][value] = cart(sub_X, sub_y, sub_features)

    # Trả về cây đã xây dựng
    return tree


# Hàm vẽ cây quyết định
def visualize_tree(tree, name):
    # Tạo đối tượng đồ thị với tên là 'name'
    dot = graphviz.Digraph(comment=name)
    # Thiết lập hướng xếp hạng từ trên xuống dưới
    dot.attr(rankdir='TB')

    # Hàm đệ quy để thêm các nút và cạnh vào đồ thị
    def add_nodes_edges(node, parent=None):
        # Kiểm tra nếu nút là một từ điển
        if isinstance(node, dict):
            # Duyệt qua từng cặp khóa-giá trị trong từ điển
            for key, value in node.items():
                # Nếu có nút cha, thêm cạnh từ cha đến khóa
                if parent:
                    dot.edge(parent, key)
                # Gọi đệ quy để xử lý giá trị
                add_nodes_edges(value, key)
        else:
            # Nếu không phải từ điển, thêm cạnh từ cha đến nút
            if parent:
                dot.edge(parent, str(node))
            # Thêm nút vào đồ thị với hình dạng hộp
            dot.node(str(node), str(node), shape='box')

    # Bắt đầu thêm nút và cạnh từ gốc cây
    add_nodes_edges(tree)
    # Xuất đồ thị thành file PNG
    dot.render(name, format='png', cleanup=True)
    # In thông báo rằng tệp PNG đã được tạo
    print(f"{name}.png has been generated.")

# Chạy các thuật toán và vẽ cây quyết định
features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
X = df[features] # Dữ liệu đầu vào
y = df['Play'] # Nhãn mục tiêu

id3_tree = id3(X, y, features)
c45_tree = c45(X, y, features)
cart_tree = cart(X, y, features)

visualize_tree(id3_tree, 'id3_tree')
visualize_tree(c45_tree, 'c45_tree')
visualize_tree(cart_tree, 'cart_tree')

print("Trees have been visualized. Check the generated PNG files.")

# Giải thích chi tiết về các thuật toán

"""
Giải thích chi tiết về các thuật toán ID3, C4.5 và CART

1. Thuật toán ID3 (Iterative Dichotomiser 3):
   
   Bước 1: Tính entropy của biến mục tiêu (trong trường hợp này là 'Play').
   Bước 2: Tính information gain cho mỗi đặc trưng.
   Bước 3: Chọn đặc trưng có information gain cao nhất làm nút gốc.
   Bước 4: Tạo các nút con cho mỗi giá trị duy nhất của đặc trưng đã chọn.
   Bước 5: Lặp lại bước 1-4 cho mỗi nút con cho đến khi đạt một trong các điều kiện dừng:
           - Tất cả các mẫu trong một nút thuộc cùng một lớp
           - Không còn đặc trưng nào để chia
           - Nút không có mẫu nào

   Khái niệm chính: Information Gain = Entropy(cha) - Tổng có trọng số của Entropy(con)

2. Thuật toán C4.5 (cải tiến từ ID3):

   Bước 1-4: Giống như ID3
   Bước 5: Tính split information cho mỗi đặc trưng.
   Bước 6: Tính gain ratio cho mỗi đặc trưng (Information Gain / Split Information).
   Bước 7: Chọn đặc trưng có gain ratio cao nhất làm nút gốc.
   Bước 8: Lặp lại bước 1-7 cho mỗi nút con cho đến khi đạt điều kiện dừng.

   Khái niệm chính: Gain Ratio = Information Gain / Split Information
   Điều này giúp giải quyết vấn đề ID3 thiên vị với các đặc trưng có nhiều giá trị duy nhất.

3. Thuật toán CART (Classification and Regression Trees):

   Bước 1: Tính chỉ số Gini cho biến mục tiêu.
   Bước 2: Với mỗi đặc trưng, tính chỉ số Gini cho mỗi điểm chia có thể.
   Bước 3: Chọn đặc trưng và điểm chia cho chỉ số Gini thấp nhất.
   Bước 4: Tạo các nút con dựa trên điểm chia đã chọn.
   Bước 5: Lặp lại bước 1-4 cho mỗi nút con cho đến khi đạt điều kiện dừng.

   Khái niệm chính: Chỉ số Gini = 1 - Tổng(p_i^2), trong đó p_i là xác suất của lớp i.
   Chỉ số Gini thấp hơn chỉ ra độ tinh khiết tốt hơn của các nút.

Sự khác biệt chính:
1. Tiêu chí chia:
   - ID3 sử dụng Information Gain
   - C4.5 sử dụng Gain Ratio
   - CART sử dụng chỉ số Gini

2. Loại đặc trưng:
   - ID3 và C4.5 hoạt động tốt với đặc trưng phân loại
   - CART có thể xử lý cả đặc trưng phân loại và số

3. Cấu trúc cây:
   - ID3 và C4.5 có thể tạo ra các phân chia đa chiều
   - CART chỉ tạo ra các phân chia nhị phân

4. Xử lý giá trị thiếu:
   - C4.5 có phương pháp tích hợp để xử lý giá trị thiếu
   - ID3 và CART thường yêu cầu tiền xử lý để xử lý giá trị thiếu

5. Phòng chống overfitting:
   - C4.5 bao gồm bước cắt tỉa để giảm overfitting
   - CART sử dụng cắt tỉa dựa trên độ phức tạp chi phí
   - ID3 cơ bản không bao gồm cắt tỉa (mặc dù có các phiên bản mở rộng)

Các thuật toán này tạo nền tảng cho việc học cây quyết định trong máy học
và rất quan trọng trong việc hiểu các phương pháp ensemble nâng cao hơn như Random Forests
và Gradient Boosting Machines.
"""