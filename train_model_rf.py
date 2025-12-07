import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# 尝试导入openpyxl用于Excel文件操作
try:
    import openpyxl
except ImportError:
    print("openpyxl未安装，将无法生成Excel文件")

# 新增三个参数控制输出类型
output_xlsx = 1  # 设置为1则输出Excel文件，否则不输出
output_csv = 0   # 设置为1则输出CSV文件，否则不输出
output_png = 0   # 设置为1则输出PNG图像文件，否则不输出

# 新增参数控制网格搜索
use_grid_search = 0  # 1-启用网格搜索 0-禁用

# 读取数据，跳过第一行标题行
input_file = r'D:\Study\QoE_QoS2\论文写作\实验程序\data2\1min_features\label3_data2_ACDFG10H_1min.csv'
data = pd.read_csv(input_file, header=0, index_col=False)  # 新增index_col=False参数
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 定义训练集比例
train_ratio = 0.8  # 训练集占比
train_pct = int(train_ratio * 100)
test_pct = int((1 - train_ratio) * 100)
split_ratio_str = f"{train_pct//10}{10-(train_pct//10)}"  # 取十位数作为文件名标识
output_file = rf'D:\Study\QoE_QoS2\论文写作\实验程序\data2\1min_features\无用特征results\rf_{current_time}_{split_ratio_str}{input_file[48:]}.csv'
output_xlsx_file = rf'D:\Study\QoE_QoS2\论文写作\实验程序\data2\1min_features\无用特征results\rf_{current_time}_{split_ratio_str}{input_file[48:]}.xlsx'

# 检查并填充空值
for col in data.columns:
    if data[col].isnull().any():
        # 获取空值所在行的行号（从0开始）
        null_rows = data[data[col].isnull()].index.tolist()
        # 将列转换为数值类型，忽略非数值数据
        data[col] = pd.to_numeric(data[col], errors='coerce')
        # 用列的平均值填充空值
        col_mean = data[col].mean()
        data[col].fillna(col_mean, inplace=True)
        # 输出列名和空值所在行的行号
        print(f"Column '{col}' contains NaN values. Filled with mean value {col_mean:.3f}. Rows with NaN: {null_rows}")

# 分离特征和标签，跳过第一列（标签）和第二列（时间戳）
X = data.iloc[:, 2:]  # 从第三列开始是特征
y = data.iloc[:, 0]   # 确保标签为浮点类型

# 按照时间顺序分割数据集，根据train_ratio参数分割训练集和测试集
split_index = int(len(data) * train_ratio)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 初始化随机森林回归模型的基础参数
base_model_params = {
    'n_estimators': 500,
    'max_depth': 3,
    'min_samples_split': 2,
    'min_samples_leaf': 4,
    'max_features': 0.6,
    'n_jobs': 32
}

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 6, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [0.2, 0.6, 'sqrt'],
}

# 根据参数决定是否进行网格搜索
if use_grid_search:
    # 使用固定random_state=1进行网格搜索
    model = RandomForestRegressor(**base_model_params)
    start_time = time.time()  # 开始计时
    
    # 计算参数组合总数
    from itertools import product
    param_combinations = list(product(*[param_grid[key] for key in param_grid.keys()]))
    total_combinations = len(param_combinations)
    print(f"Starting grid search with {total_combinations} parameter combinations...")
    
    # 手动实现网格搜索以跟踪进度
    best_score = float('-inf')
    best_params = None
    results = []
    
    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_grid.keys(), params))
        elapsed_time = time.time() - start_time
        avg_time_per_combination = elapsed_time / (i + 1) if i > 0 else 0
        estimated_remaining_time = avg_time_per_combination * (total_combinations - i - 1)
        
        print(f"Testing combination {i+1}/{total_combinations}: {param_dict}")
        print(f"Elapsed time: {elapsed_time:.1f}s. Estimated remaining time: {estimated_remaining_time:.1f}s")
        
        # 创建并训练模型
        # 合并基础参数和网格搜索参数，param_dict中的值会覆盖base_model_params中的同名参数
        merged_params = base_model_params.copy()
        merged_params.update(param_dict)
        test_model = RandomForestRegressor(**merged_params)
        
        # 简化的交叉验证（3折）
        scores = []
        fold_size = len(X_train) // 3
        for fold in range(3):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < 2 else len(X_train)
            
            X_val_fold = X_train.iloc[val_start:val_end]
            y_val_fold = y_train.iloc[val_start:val_end]
            X_train_fold = pd.concat([X_train.iloc[:val_start], X_train.iloc[val_end:]])
            y_train_fold = pd.concat([y_train.iloc[:val_start], y_train.iloc[val_end:]])
            
            test_model.fit(X_train_fold, y_train_fold)
            score = test_model.score(X_val_fold, y_val_fold)
            scores.append(score)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        results.append({
            'params': param_dict,
            'mean_score': mean_score,
            'std_score': std_score
        })
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = param_dict
            
        print(f"Current score: {mean_score:.6f} (+/- {std_score*2:.6f})")
        print(f"Best score so far: {best_score:.6f} with parameters: {best_params}")
        print("-" * 50)
    
    # 输出最优参数组合
    print("=" * 70)
    print("Grid search completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.6f}")
    print("=" * 70)
    
    # 使用最佳参数创建最终模型
    # 合并基础参数和最佳参数，best_params中的值会覆盖base_model_params中的同名参数
    merged_best_params = base_model_params.copy()
    merged_best_params.update(best_params)
    best_model = RandomForestRegressor(**merged_best_params)
    best_model.fit(X_train, y_train)
    training_time = (time.time() - start_time) * 1000  # 计算训练时间并转换为毫秒
    
    # 记录单次运行结果
    start_time = time.time()  # 开始计时测试时间
    y_pred = best_model.predict(X_test).round(3)
    testing_time = (time.time() - start_time) * 1000  # 计算测试时间并转换为毫秒
    y_train_pred = best_model.predict(X_train).round(3)
    
    # 创建训练集的DataFrame并计算百分比差异（添加极小值防止除以0）
    train_results = pd.DataFrame({
        'True Label': y_train,  # 确保浮点类型
        'Predicted Label': y_train_pred  
    })
    train_results['True Label'] = train_results['True Label'].replace(0, 1e-6)  # 新增：统一替换0值
    train_results['Absolute Percentage Difference (%)'] = (
        (abs(train_results['True Label'] - train_results['Predicted Label']) / 
         train_results['True Label']) * 100
    ).round(3)

    # 新增：将测试集标签和预测值保存到CSV文件中，并计算绝对百分比差距
    results_df = pd.DataFrame({
        'True Label': y_test, 
        'Predicted Label': y_pred  # 使用已四舍五入的预测值
    })
    # 修改：保留绝对百分比差距三位小数
    results_df['Absolute Percentage Difference (%)'] = (
        (abs(results_df['True Label'] - results_df['Predicted Label']) / results_df['True Label']) * 100
    ).round(3)

    # 计算评估指标
    # 训练集指标
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rad = train_results['Absolute Percentage Difference (%)'].mean()

    # 测试集指标
    test_mae = mean_absolute_error(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    test_rad = results_df['Absolute Percentage Difference (%)'].mean()
    
    # 存储单次运行的结果
    all_runs_metrics = [{
        'random_state': 1,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rad': train_rad,
        'test_rad': test_rad,
        'training_time': training_time,
        'testing_time': testing_time
    }]
    
    # 选择最佳模型作为最终模型
    final_model = best_model
    final_y_pred = y_pred
    final_y_train_pred = y_train_pred
    final_results = results_df
    final_train_results = train_results
    
else:
    # 不使用网格搜索，进行10次不同random_state的训练和测试
    all_runs_metrics = []
    
    for random_state in range(10):
        print(f"Running experiment with random_state={random_state}")
        
        # 初始化随机森林回归模型
        model = RandomForestRegressor(random_state=random_state, **base_model_params)
        
        start_time = time.time()  # 开始计时
        model.fit(X_train, y_train)
        training_time = (time.time() - start_time) * 1000  # 计算训练时间并转换为毫秒
        
        # 测试模型
        start_time = time.time()  # 开始计时测试时间
        y_pred = model.predict(X_test).round(3)
        testing_time = (time.time() - start_time) * 1000  # 计算测试时间并转换为毫秒
        y_train_pred = model.predict(X_train).round(3)
        
        # 计算评估指标
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_rad = np.mean(np.abs(y_train - y_train_pred) / np.where(y_train != 0, y_train, 1e-6)) * 100
        
        test_mae = mean_absolute_error(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        test_rad = np.mean(np.abs(y_test - y_pred) / np.where(y_test != 0, y_test, 1e-6)) * 100
        
        # 存储本次运行的结果
        run_metrics = {
            'random_state': random_state,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rad': train_rad,
            'test_rad': test_rad,
            'training_time': training_time,
            'testing_time': testing_time
        }
        all_runs_metrics.append(run_metrics)
        
        # 如果是random_state=0，保存详细结果
        if random_state == 0:
            # 创建训练集的DataFrame并计算百分比差异（添加极小值防止除以0）
            train_results = pd.DataFrame({
                'True Label': y_train,  # 确保浮点类型
                'Predicted Label': y_train_pred  
            })
            train_results['True Label'] = train_results['True Label'].replace(0, 1e-6)  # 新增：统一替换0值
            train_results['Absolute Percentage Difference (%)'] = (
                (abs(train_results['True Label'] - train_results['Predicted Label']) / 
                 train_results['True Label']) * 100
            ).round(3)

            # 新增：将测试集标签和预测值保存到CSV文件中，并计算绝对百分比差距
            results_df = pd.DataFrame({
                'True Label': y_test, 
                'Predicted Label': y_pred  # 使用已四舍五入的预测值
            })
            # 修改：保留绝对百分比差距三位小数
            results_df['Absolute Percentage Difference (%)'] = (
                (abs(results_df['True Label'] - results_df['Predicted Label']) / results_df['True Label']) * 100
            ).round(3)
            
            # 保存random_state=0的结果作为最终结果
            final_model = model
            final_y_pred = y_pred
            final_y_train_pred = y_train_pred
            final_results = results_df
            final_train_results = train_results
    
    # 计算平均指标
    avg_metrics = {}
    metric_keys = ['train_mae', 'test_mae', 'train_mse', 'test_mse', 'train_r2', 'test_r2', 'train_rad', 'test_rad', 'training_time', 'testing_time']
    for key in metric_keys:
        avg_metrics[key] = np.mean([run[key] for run in all_runs_metrics])
    
    # 添加平均指标到结果列表
    avg_run_metrics = {'random_state': 'Average'}
    for key in metric_keys:
        avg_run_metrics[key] = avg_metrics[key]
    all_runs_metrics.append(avg_run_metrics)

# 使用最终模型的结果进行后续处理
y_pred = final_y_pred
y_train_pred = final_y_train_pred
results = final_results
train_results = final_train_results

# 计算评估指标
# 训练集指标
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
train_rad = train_results['Absolute Percentage Difference (%)'].mean()

# 测试集指标
test_mae = mean_absolute_error(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)
test_rad = results['Absolute Percentage Difference (%)'].mean()

# 创建性能指标DataFrame
metrics_df = pd.DataFrame({
    'MAE': [round(train_mae, 2), round(test_mae, 2)],
    'MSE': [round(train_mse, 2), round(test_mse, 2)],
    'R2': [round(train_r2, 2), round(test_r2, 2)],
    'RAD': [f"{round(train_rad, 2)}%", f"{round(test_rad, 2)}%"]
}, index=['train', 'test'])

# 在性能指标DataFrame中添加训练和测试时间
if use_grid_search:
    metrics_df['Training Time (ms)'] = [training_time, '']
    metrics_df['Testing Time (ms)'] = ['', testing_time]
else:
    # 添加所有运行的平均时间
    avg_training_time = np.mean([run['training_time'] for run in all_runs_metrics[:-1]])  # 排除平均行
    avg_testing_time = np.mean([run['testing_time'] for run in all_runs_metrics[:-1]])   # 排除平均行
    metrics_df['Training Time (ms)'] = [avg_training_time, '']
    metrics_df['Testing Time (ms)'] = ['', avg_testing_time]

# 创建详细数据DataFrame
train_detail_df = pd.DataFrame({
    'Timestamp': range(len(y_train)),
    'True Label': y_train.values,
    'Predicted Label': y_train_pred
})

test_detail_df = pd.DataFrame({
    'Timestamp': range(len(y_test)),
    'True Label': y_test.values,
    'Predicted Label': y_pred
})

# 创建所有运行结果的DataFrame
all_runs_df = pd.DataFrame(all_runs_metrics)

# 写入参数到文件开头
if output_csv:
    with open(output_file, 'w') as f:
        if use_grid_search:
            f.write(f"Best Parameters from GridSearch: {best_params if use_grid_search else 'N/A'}\n")
        else:
            f.write("Random Forest Parameters: n_estimators=300, max_depth=3, min_samples_split=2, min_samples_leaf=1\n")
            f.write("Performed 10 runs with random_state from 0 to 9\n")
        f.write("--------------------------------------------------\n")

# 新增写入测试集结果：
if output_csv:
    with open(output_file, 'a') as f:
        f.write("\nTest Set Results (random_state=0):\n")
        f.write(results.to_string(index=False))
        f.write("\n--------------------------------------------------\n")

# 创建训练集的DataFrame并计算百分比差异
train_results = pd.DataFrame({
    'True Label': y_train, 
    'Predicted Label': y_train_pred  # 使用已四舍五入的预测值
})
train_results['Absolute Percentage Difference (%)'] = (
    (abs(train_results['True Label'] - train_results['Predicted Label']) / train_results['True Label']) * 100
).round(3)
# 修改：保留三位小数并添加分隔线
if output_csv:
    with open(output_file, 'a') as f:
        mean_k = results['Absolute Percentage Difference (%)'].mean()
        f.write(f"\n--------------------------------------------------\n")
        f.write(f"Mean Absolute Percentage Difference (%): {mean_k:.3f}")

if output_csv:
    with open(output_file, 'a') as f:
        # 此时 train_results 已定义，可安全访问
        mean_train_k = train_results['Absolute Percentage Difference (%)'].mean()
        f.write(f"\n--------------------------------------------------\n")
        f.write(f"Mean Absolute Percentage Difference on Training Set (%): {mean_train_k:.3f}")

# 新增：绘制预测值与标签差距的线性图
if output_png:
    plt.figure(figsize=(20, 6))
    plt.plot(y_test.values, label='True Label')
    plt.plot(y_pred, label='Predicted Label')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Comparison of True Label and Predicted Label')
    plt.legend()
    plt.savefig(output_xlsx_file.replace('.xlsx', '.png'))
    #plt.show()

# 保存为Excel文件
if output_xlsx:
    with pd.ExcelWriter(output_xlsx_file, engine='openpyxl') as writer:
        # 写入性能指标表
        metrics_df.to_excel(writer, sheet_name='Metrics', index=True)
        
        # 写入详细数据（用空列分隔）
        train_detail_df.to_excel(writer, sheet_name='Detailed Data', startcol=0, index=False)
        test_detail_df.to_excel(writer, sheet_name='Detailed Data', startcol=5, index=False)
        
        # 写入所有运行结果
        all_runs_df.to_excel(writer, sheet_name='All Runs Results', index=False)