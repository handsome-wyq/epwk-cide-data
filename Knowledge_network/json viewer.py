'''import json

# 文件路径

json_file_path = "D:/Documents/Desktop/ner/420test_output/420test_output.json"
# 读取并打印 JSON 文件内容
try:
    with open(json_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 跳过空行
                try:
                    json_obj = json.loads(line)
                    print("JSON 行内容:")
                    print(json.dumps(json_obj, ensure_ascii=False, indent=2))
                    print("-" * 50)
                except json.JSONDecodeError as e:
                    print(f"解析错误在行: {line}, 错误: {e}")
except FileNotFoundError:
    print(f"文件 {json_file_path} 不存在，请检查路径！")
except Exception as e:
    print(f"读取文件时发生错误: {str(e)}")'''
'''import json

json_file_path = "D:/Documents/Desktop/ner/422Experiment/汽车.json"

try:
    with open(json_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    json_obj = json.loads(line)

                    # 1. 输出标题和需求文本
                    title = json_obj.get("标题", "无标题")
                    requirement = json_obj.get("需求文本", "无需求文本")
                    print(f"标题: {title}")
                    print(f"需求文本: {requirement}\n")

                    # 2. 提取并输出设计对象
                    entities = json_obj.get('提取实体', {}).get('structured_data', {}).get('entities', [])
                    design_objects = [e['id'] for e in entities if e.get('type') == '设计对象']

                    if design_objects:
                        print("设计对象:")
                        for obj in design_objects:
                            print(f"- {obj}")
                    else:
                        print("未找到设计对象")


                    print("\n" + "=" * 50 + "\n")  # 分隔线
                except (json.JSONDecodeError, KeyError, Exception) as e:
                    continue  # 跳过错误行
except FileNotFoundError:
    print(f"文件 {json_file_path} 不存在")
except Exception as e:
    pass ''' # 静默处理其他错误
#查看json结构
import json

json_file_path = "D:\Documents\Desktop\Crowd_intelligence_wyq\抽取结果\817第二批\开发\开发-招标.json"


def extract_entities(entities, target_type):
    """通用实体提取函数"""
    results = []
    for e in entities:
        try:
            if e.get('type') == target_type:
                # 处理可能缺少id字段的情况
                entity_id = e.get('id', '未知ID')
                # 提取属性（如果有）
                attributes = e.get('attributes', {})
                results.append((entity_id, attributes))
        except (KeyError, AttributeError):
            continue
    return results


try:
    with open(json_file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    json_obj = json.loads(line)

                    # 基础信息输出
                    print(f"\n{'=' * 50}")
                    print(f"记录 #{line_num}")
                    print(f"标题: {json_obj.get('标题', '无标题')}")
                    print(f"需求文本: {json_obj.get('需求文本', '无需求文本')}")

                    # 实体提取
                    entities = json_obj.get('提取实体', {}).get('structured_data', {}).get('entities', [])
                    id_text = ' '.join(item['id'] for item in entities)
                    print(id_text)

                    ''''# 1. 设计对象提取
                    design_objects = extract_entities(entities, '设计对象')
                    #print("\n[设计对象]")
                    if design_objects:
                        for idx, (obj_id, attrs) in enumerate(design_objects, 1):
                            attr_str = ", ".join([f"{k}:{v}" for k, v in attrs.items()]) if attrs else "无属性"
                            print(f" {obj_id} ")
                    else:
                        print("未找到设计对象")'''



                    #print("=" * 50)

                except json.JSONDecodeError as e:
                    print(f"\n记录 #{line_num} JSON解析错误: {str(e)}")
                except Exception as e:
                    print(f"\n记录 #{line_num} 处理错误: {str(e)}")

except FileNotFoundError:
    print(f"文件 {json_file_path} 不存在")
except Exception as e:
    print(f"发生未预期错误: {str(e)}")