import json

# 打开文件并读取 JSON 数据
with open('/GPFS/data/gjliu/Auto-driving/V2Xverse/simulation/leaderboard/data/scenarios/town05_all_scenarios.json', 'r') as file:
    data = json.load(file)
    new_data = {'available_scenarios':[{'Town05':[]}]}
    for scenarios in data['available_scenarios'][0]['Town05']:
        if scenarios['scenario_type'] in ['Scenario1','Scenario3','Scenario4']:
            new_data['available_scenarios'][0]['Town05'].append(scenarios)
    with open('/GPFS/data/gjliu/Auto-driving/V2Xverse/simulation/leaderboard/data/scenarios/town05_all_scenarios_2.json', 'w') as file2:
        json.dump(new_data, file2, indent=2)
