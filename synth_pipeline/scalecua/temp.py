# import json
# data_name = "250407"

# for path in [f"/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/{data_name}/{data_name}_sc.jsonl",
# f"/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/{data_name}/{data_name}_instmod.jsonl",
# f"/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/{data_name}/{data_name}_venuspred.jsonl",
# f"/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/{data_name}/{data_name}_category.jsonl",
# ]:

#     with open(path, 'r', encoding='utf-8') as f:
#         data = [json.loads(line) for line in f]
#         print(len(data))




import json
for data_name in [
    "250310",
    "250317",
    "250328",
    "250407",
    "250414",
    "250421",
    "250428",
    "250505",
    "250526",
    "250623",
    "250630",
    "250707",
    "250714"
]:


    path = f"/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/{data_name}/{data_name}_sc.jsonl"
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        print(f"{data_name}",len(data))