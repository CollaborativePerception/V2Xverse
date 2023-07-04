from lxml import etree
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree, Element, SubElement, tostring
import os
import numpy as np
import re
from tqdm import tqdm
import shutil

def update_xml():
    #查找节点并更新
    route_order = np.zeros(20)
    pattern = re.compile('.*town(\d\d)')
    file_dir = '/GPFS/data/gjliu/Auto-driving/Cop3/leaderboard_lgj/data/evaluation_routes/gjliu'
    save_dir = '/GPFS/data/gjliu/Auto-driving/Cop3/leaderboard_lgj/data/evaluation_routes/gjliu/split'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    print("refresh succeed")

    for file in tqdm(os.listdir(file_dir)):
        if not file.endswith('xml'):
            continue
        if file.endswith('tiny.xml'):
            continue
        res = pattern.findall(file)
        town = res[0]
        # print(file,'  ',town)

        tree = ET.parse(os.path.join(file_dir, file))
        root = tree.getroot()
        id_list = []
        for node in tree.findall("route"):
            id_list.append(node.attrib["id"])
        for route_id in id_list:
            tree = ET.parse(os.path.join(file_dir, file))
            root = tree.getroot()
            for node in tree.findall("route"):
                if node.attrib["id"] != route_id:
                    root.remove(node)
                else:
                    node.attrib["id"] = str(int(route_order[int(town)]))
                # node.tag = "path"
            # ET.dump(tree)  #打印xml
            tree.write(os.path.join(save_dir, 'town{}_short_r{}.xml'.format(town,int(route_order[int(town)])+300)), encoding="utf-8", xml_declaration=True)
            route_order[int(town)] += 1

update_xml()


