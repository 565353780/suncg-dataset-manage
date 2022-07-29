import json
import cv2
import numpy as np
import os
import csv
import random
from time import time

class Node:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, label):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.label = label

        self.id = None

        self.trans_x_min = None
        self.trans_x_max = None
        self.trans_y_min = None
        self.trans_y_max = None
        self.trans_z_min = None
        self.trans_z_max = None

    def update_trans(self, scale_x, scale_y, trans):
        self.trans_x_min = int(self.x_min * scale_x) + trans[0]
        self.trans_x_max = int(self.x_max * scale_x) + trans[0]
        self.trans_y_min = int(self.y_min * scale_y) + trans[1]
        self.trans_y_max = int(self.y_max * scale_y) + trans[1]
        self.trans_z_min = self.z_min + trans[2]
        self.trans_z_max = self.z_max + trans[2]

class Room:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, target_width, target_height, size_fixed):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.target_width = target_width
        self.target_height = target_height
        self.size_fixed = size_fixed

        self.id = None

        self.trans_x_min = None
        self.trans_x_max = None
        self.trans_y_min = None
        self.trans_y_max = None
        self.trans_z_min = None
        self.trans_z_max = None

        self.nodes_z_min = None
        self.node_list = []
        self.scale_x = None
        self.scale_y = None
        self.trans = None
        self.width = None
        self.height = None

        self.is_empty = True
        self.is_valid = False

        self.label_channel = None

    def add_node(self, node):
        node.id = self.id + "_" + str(len(self.node_list))
        self.node_list.append(node)
        if self.nodes_z_min is None:
            self.nodes_z_min = node.z_min
        elif node.z_min < self.nodes_z_min:
            self.nodes_z_min = node.z_min

        if self.is_empty:
            self.is_empty = False

    def compute_trans(self):
        self.trans_x_min = int(self.x_min * self.scale_x) + self.trans[0]
        self.trans_x_max = int(self.x_max * self.scale_x) + self.trans[0]
        self.trans_y_min = int(self.y_min * self.scale_y) + self.trans[1]
        self.trans_y_max = int(self.y_max * self.scale_y) + self.trans[1]
        self.trans_z_min = self.z_min + self.trans[2]
        self.trans_z_max = self.z_max + self.trans[2]

    def update_trans(self, trans_root, scale_x, scale_y, trans):
        if not self.is_empty:
            if trans_root == "House" or trans_root == "Level":
                self.scale_x = scale_x
                self.scale_y = scale_y
                self.trans = trans
                self.compute_trans()
            elif trans_root == "Room":
                self.scale_x = self.target_width / (self.x_max - self.x_min)
                self.scale_y = self.target_height / (self.y_max - self.y_min)

                if not self.size_fixed:
                    if self.scale_y < self.scale_x:
                        self.scale_x = self.scale_y
                    else:
                        self.scale_y = self.scale_x

                    self.width = int((self.x_max - self.x_min) * self.scale_x)
                    self.height = int((self.y_max - self.y_min) * self.scale_y)
                else:
                    self.width = self.target_width
                    self.height = self.target_height

                self.trans = [- int(self.x_min * self.scale_x), - int(self.y_min * self.scale_y), - self.nodes_z_min]

                self.compute_trans()

            for node in self.node_list:
                node.update_trans(self.scale_x, self.scale_y, self.trans)

    def create_label_channel(self, label):
        self.is_valid = False

        self.update_trans("Room", None, None, None)

        if not self.is_empty:
            if label.channel_num == 1:
                self.label_channel = np.zeros((self.height, self.width), label.number_type)
            else:
                self.label_channel = np.zeros((self.height, self.width, label.channel_num), label.number_type)

            valid_node_num = 0

            for node in self.node_list:
                label_index = label.get_label_index(node.label)
                if label_index is not None:
                    cv2.rectangle(self.label_channel, (node.trans_x_min, node.trans_y_min), (node.trans_x_max, node.trans_y_max), label.color_list[label_index], -1)
                    valid_node_num += 1

            if valid_node_num >= label.min_node_num:
                self.is_valid = True

        if self.is_valid:
            label.save_label_channel("Room", self.label_channel, self.id)

        if label.free_label:
            self.label_channel = None

class Level:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, target_width, target_height, size_fixed):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.target_width = target_width
        self.target_height = target_height
        self.size_fixed = size_fixed

        self.id = None

        self.trans_x_min = None
        self.trans_x_max = None
        self.trans_y_min = None
        self.trans_y_max = None
        self.trans_z_min = None
        self.trans_z_max = None

        self.rooms_z_min = None
        self.room_list = []
        self.scale_x = None
        self.scale_y = None
        self.trans = None
        self.width = None
        self.height = None

        self.is_empty = True
        self.is_valid = False

        self.label_channel = None

    def add_node(self, node):
        for room in self.room_list:
            if node.x_min >= room.x_min and node.x_max <= room.x_max and node.y_min >= room.y_min and node.y_max <= room.y_max and node.z_min >= room.z_min and node.z_max <= room.z_max:
                room.add_node(node)

                if self.is_empty:
                    self.is_empty = False
                if room.is_empty:
                    room.is_empty = False

                return

    def add_room(self, room):
        room.id = self.id + "_" + str(len(self.room_list))
        self.room_list.append(room)
        if self.rooms_z_min is None:
            self.rooms_z_min = room.z_min
        elif room.z_min < self.rooms_z_min:
            self.rooms_z_min = room.z_min

    def compute_trans(self):
        self.trans_x_min = int(self.x_min * self.scale_x) + self.trans[0]
        self.trans_x_max = int(self.x_max * self.scale_x) + self.trans[0]
        self.trans_y_min = int(self.y_min * self.scale_y) + self.trans[1]
        self.trans_y_max = int(self.y_max * self.scale_y) + self.trans[1]
        self.trans_z_min = self.z_min + self.trans[2]
        self.trans_z_max = self.z_max + self.trans[2]

    def update_trans(self, trans_root, scale_x, scale_y, trans):
        if not self.is_empty:
            if trans_root == "House":
                self.scale_x = scale_x
                self.scale_y = scale_y
                self.trans = trans
                self.compute_trans()
            elif trans_root == "Level":
                self.scale_x = self.target_width / (self.x_max - self.x_min)
                self.scale_y = self.target_height / (self.y_max - self.y_min)

                if not self.size_fixed:
                    if self.scale_y < self.scale_x:
                        self.scale_x = self.scale_y
                    else:
                        self.scale_y = self.scale_x

                    self.width = int((self.x_max - self.x_min) * self.scale_x)
                    self.height = int((self.y_max - self.y_min) * self.scale_y)
                else:
                    self.width = self.target_width
                    self.height = self.target_height

                self.trans = [- int(self.x_min * self.scale_x), - int(self.y_min * self.scale_y), - self.rooms_z_min]

                self.compute_trans()

            for room in self.room_list:
                room.update_trans(trans_root, self.scale_x, self.scale_y, self.trans)

    def create_label_channel(self, label):
        self.is_valid = False

        self.update_trans("Level", None, None, None)

        if not self.is_empty:
            if label.channel_num == 1:
                self.label_channel = np.zeros((self.height, self.width), label.number_type)
            else:
                self.label_channel = np.zeros((self.height, self.width, label.channel_num), label.number_type)

            valid_node_num = 0

            for room in self.room_list:
                for node in room.node_list:
                    label_index = label.get_label_index(node.label)
                    if label_index is not None:
                        cv2.rectangle(self.label_channel, (node.trans_x_min, node.trans_y_min), (node.trans_x_max, node.trans_y_max), label.color_list[label_index], -1)
                        valid_node_num += 1

            if valid_node_num >= label.min_node_num:
                self.is_valid = True

            for room in self.room_list:
                room.create_label_channel(label)

        if self.is_valid:
            label.save_label_channel("Level", self.label_channel, self.id)

        if label.free_label:
            self.label_channel = None


class House:
    def __init__(self, json_id, target_width, target_height, size_fixed):
        self.json_id = json_id
        self.target_width = target_width
        self.target_height = target_height
        self.size_fixed = size_fixed

        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.z_min = None
        self.z_max = None

        self.trans_x_min = None
        self.trans_x_max = None
        self.trans_y_min = None
        self.trans_y_max = None
        self.trans_z_min = None
        self.trans_z_max = None

        self.levels_z_min = None
        self.level_list = []
        self.scale_x = None
        self.scale_y = None
        self.trans = None
        self.width = None
        self.height = None

        self.x_index = None
        self.y_index = None
        self.z_index = None

        self.json = None

        self.is_empty = True
        self.is_valid = False

        self.label_channel = None

    def add_node(self, node):
        for level in self.level_list:
            for room in level.room_list:
                if node.x_min >= room.x_min and node.x_max <= room.x_max and node.y_min >= room.y_min and node.y_max <= room.y_max and node.z_min >= room.z_min and node.z_max <= room.z_max:
                    room.add_node(node)

                    if self.is_empty:
                        self.is_empty = False
                    if room.is_empty:
                        room.is_empty = False
                    if level.is_empty:
                        level.is_empty = False

                    return

    def add_room(self, room):
        for level in self.level_list:
            if room.x_min >= level.x_min and room.x_max <= level.x_max and room.y_min >= level.y_min and room.y_max <= level.y_max and room.z_min >= level.z_min and room.z_max <= level.z_max:
                level.add_room(room)

                return

    def add_level(self, level):
        level.id = self.json_id + "_" + str(len(self.level_list))
        self.level_list.append(level)
        if self.levels_z_min is None:
            self.levels_z_min = level.z_min
        elif level.z_min < self.levels_z_min:
            self.levels_z_min = level.z_min

    def compute_trans(self):
        self.trans_x_min = int(self.x_min * self.scale_x) + self.trans[0]
        self.trans_x_max = int(self.x_max * self.scale_x) + self.trans[0]
        self.trans_y_min = int(self.y_min * self.scale_y) + self.trans[1]
        self.trans_y_max = int(self.y_max * self.scale_y) + self.trans[1]
        self.trans_z_min = self.z_min + self.trans[2]
        self.trans_z_max = self.z_max + self.trans[2]

    def update_trans(self, trans_root):
        if not self.is_empty:
            if trans_root == "House":
                self.scale_x = self.target_width / (self.x_max - self.x_min)
                self.scale_y = self.target_height / (self.y_max - self.y_min)

                if not self.size_fixed:
                    if self.scale_y < self.scale_x:
                        self.scale_x = self.scale_y
                    else:
                        self.scale_y = self.scale_x

                    self.width = int((self.x_max - self.x_min) * self.scale_x)
                    self.height = int((self.y_max - self.y_min) * self.scale_y)
                else:
                    self.width = self.target_width
                    self.height = self.target_height

                self.trans = [- int(self.x_min * self.scale_x), - int(self.y_min * self.scale_y), - self.levels_z_min]

                self.compute_trans()

            for level in self.level_list:
                level.update_trans(trans_root, self.scale_x, self.scale_y, self.trans)

    def load_bbox(self, json_dict):
        x_min = json_dict["bbox"]["min"][self.x_index]
        x_max = json_dict["bbox"]["max"][self.x_index]

        y_min = json_dict["bbox"]["min"][self.y_index]
        y_max = json_dict["bbox"]["max"][self.y_index]

        z_min = json_dict["bbox"]["min"][self.z_index]
        z_max = json_dict["bbox"]["max"][self.z_index]

        return x_min, x_max, y_min, y_max, z_min, z_max

    def load_json(self, json_file_path):
        with open(json_file_path, "r") as f:
            self.json = json.load(f)

        up = self.json["up"]

        if up[0] == 1:
            self.x_index = 1
            self.y_index = 2
            self.z_index = 0
        elif up[1] == 1:
            self.x_index = 2
            self.y_index = 0
            self.z_index = 1
        else:
            self.x_index = 0
            self.y_index = 1
            self.z_index = 2

        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = self.load_bbox(self.json)

        if "levels" in self.json:
            for level in self.json["levels"]:
                if "bbox" not in level:
                    continue

                level_x_min, level_x_max, level_y_min, level_y_max, level_z_min, level_z_max = self.load_bbox(level)

                current_level = Level(level_x_min, level_x_max, level_y_min, level_y_max, level_z_min, level_z_max, self.target_width, self.target_height, self.size_fixed)

                self.add_level(current_level)

                if "nodes" in level:
                    for node in level["nodes"]:
                        if "valid" in node and "type" in node:
                            if node["valid"] == 1 and node["type"] == "Room":

                                room_x_min, room_x_max, room_y_min, room_y_max, room_z_min, room_z_max = self.load_bbox(node)

                                current_room = Room(room_x_min, room_x_max, room_y_min, room_y_max, room_z_min, room_z_max, self.target_width, self.target_height, self.size_fixed)

                                self.add_room(current_room)

                    for node in level["nodes"]:
                        if "valid" in node and "type" in node:
                            if node["valid"] == 1 and node["type"] != "Room":
                                if "modelId" not in node:
                                    continue

                                node_x_min, node_x_max, node_y_min, node_y_max, node_z_min, node_z_max = self.load_bbox(node)

                                current_node = Node(node_x_min, node_x_max, node_y_min, node_y_max, node_z_min, node_z_max, node["modelId"])

                                self.add_node(current_node)

    def create_label_channel(self, label):
        self.is_valid = False

        self.update_trans("House")

        if not self.is_empty:
            if label.channel_num == 1:
                self.label_channel = np.zeros((self.height, self.width), label.number_type)
            else:
                self.label_channel = np.zeros((self.height, self.width, label.channel_num), label.number_type)

            valid_node_num = 0

            for level in self.level_list:
                for room in level.room_list:
                    for node in room.node_list:
                        label_index = label.get_label_index(node.label)
                        if label_index is not None:
                            cv2.rectangle(self.label_channel, (node.trans_x_min, node.trans_y_min), (node.trans_x_max, node.trans_y_max), label.color_list[label_index], -1)
                            valid_node_num += 1

            if valid_node_num >= label.min_node_num:
                self.is_valid = True

            for level in self.level_list:
                level.create_label_channel(label)

        if self.is_valid:
            label.save_label_channel("House", self.label_channel, self.json_id)

        if label.free_label:
            self.label_channel = None


class Label:
    def __init__(self,
                label_file_path,
                valid_label_list,
                is_binary,
                min_node_num,
                save_object,
                save_path,
                channel_num,
                use_color,
                free_label,
                number_type
                ):
        self.label_file_path = label_file_path
        self.valid_label_list = valid_label_list
        self.is_binary = is_binary
        self.min_node_num = min_node_num
        self.save_object = save_object
        self.save_path = save_path
        self.channel_num = channel_num
        self.use_color = use_color
        self.free_label = free_label
        self.number_type = number_type

        self.save_as_npy = False

        self.csv_data = []

        self.color_list = []

        self.label_array = []

        self.load_label()

        self.create_color()

        self.create_save_path()

    def load_label(self):
        self.csv_data = []

        with open(self.label_file_path, "r") as f:
            reader = csv.reader(f)

            for row in reader:
                self.csv_data.append(row)

    def create_color(self):
        color_black = []
        for i in range(self.channel_num):
            color_black.append(0)
        if self.channel_num == 1:
            self.color_list.append(color_black[0])
        else:
            self.color_list.append(tuple(color_black))

        for i in range(len(self.valid_label_list)):
            label_color = []
            if self.is_binary:
                if self.use_color:
                    for j in range(self.channel_num):
                        label_color.append(255)
                else:
                    for j in range(self.channel_num):
                        label_color.append(1)
            elif self.use_color:
                for j in range(self.channel_num):
                    label_color.append(random.randint(0, 255))
            else:
                for j in range(self.channel_num):
                    label_color.append(i + 1)
            if self.channel_num == 1:
                self.color_list.append(label_color[0])
            else:
                self.color_list.append(tuple(label_color))

    def create_save_path(self):
        if self.save_path[-4:] == ".npy":
            save_path_split = self.save_path.split("/")
            save_file_name = save_path_split[len(save_path_split) - 1]
            target_save_path = self.save_path.split(save_file_name)[0]
            if not os.path.exists(target_save_path):
                os.makedirs(target_save_path)
            self.save_as_npy = True
        elif not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        for object_type in self.save_object:
            if not os.path.exists(self.save_path + object_type + "/"):
                os.makedirs(self.save_path + object_type + "/")

    def is_label_valid(self, label_info):
        if self.valid_label_list is None:
            return True

        if label_info[3] in self.valid_label_list:
            return True

        return False

    def get_label_info(self, label):
        for line in self.csv_data:
            if line[1] == label:
                if self.is_label_valid(line):
                    return line

                return None

        return None

    def get_label_index(self, label):
        label_info = self.get_label_info(label)

        if label_info is None:
            return None

        label_index = 1

        for valid_label in self.valid_label_list:
            if label_info[3] == valid_label:
                return label_index

            label_index += 1

    def save_label_channel(self, object_type, label_channel, id):
        if object_type in self.save_object:
            if self.save_path is not None:
                if self.save_as_npy:
                    self.label_array.append(label_channel)
                else:
                    cv2.imwrite(self.save_path + object_type + "/" + id + ".jpg", label_channel)


class SUNCGDataBase:
    def __init__(self,
                json_id_path,
                target_width,
                target_height,
                size_fixed,
                label_file_path,
                valid_label_list=None,
                is_binary=False,
                min_node_num=1,
                save_object=[],
                save_path=None,
                channel_num=1,
                use_color=False,
                free_label=True,
                number_type=np.uint8
                ):
        self.json_id_path = json_id_path
        self.target_width = target_width
        self.target_height = target_height
        self.size_fixed = size_fixed
        self.label_file_path = label_file_path
        self.valid_label_list = valid_label_list
        self.is_binary = is_binary
        self.min_node_num = min_node_num
        self.save_object = save_object
        self.save_path = save_path
        self.channel_num = channel_num
        self.use_color = use_color
        self.free_label = free_label
        self.number_type = number_type

        self.json_id_list = []
        self.house_list = []

        if self.json_id_path[-1] != "/":
            self.json_id_path += "/"

        if self.save_path[-4:] != ".npy":
            if self.save_path[-1] != "/":
                self.save_path += "/"

        self.label = Label(self.label_file_path,
                            self.valid_label_list,
                            self.is_binary,
                            self.min_node_num,
                            self.save_object,
                            self.save_path,
                            self.channel_num,
                            self.use_color,
                            self.free_label,
                            self.number_type
                            )

        self.load_json()

    def add_house(self, json_id, json_file_path):
        house = House(json_id, self.target_width, self.target_height, self.size_fixed)
        house.load_json(json_file_path)

        house.create_label_channel(self.label)

        self.house_list.append(house)

    def load_json(self):
        if self.json_id_path is not None:
            source_json_id_list = os.listdir(self.json_id_path)

            total_json_id_num = len(source_json_id_list)

            loaded_json_id_num = 0

            for json_id in source_json_id_list:
                json_file_path = self.json_id_path + json_id + "/house.json"
                if os.path.exists(json_file_path):
                    self.json_id_list.append([json_id, json_file_path])

                    self.add_house(json_id, json_file_path)

                    loaded_json_id_num += 1

                    # # for test, or your pc will boom.
                    # if loaded_json_id_num > 10:
                    #     break

                    print("\rLoaded houses num :", loaded_json_id_num, "/", total_json_id_num, "    ", end="")

            print()

            if self.label.save_as_npy:
                np.save(self.save_path, self.label.label_array)
                print("Saved as npy file at :", self.save_path)

    def create_label_channel(self):
        for house in self.house_list:
            house.create_label_channel(self.label)

    def load_label_channel(self, trans_root, room_id_list=None):
        if room_id_list is None:
            while True:
                house_index, level_index, room_index = [None, None, None]

                if trans_root == "House":
                    len_house_list = len(self.house_list)
                    if len_house_list == 0:
                        continue
                    house_index = random.randint(0, len_house_list - 1)

                    if not self.house_list[house_index].is_valid:
                        continue

                    return self.house_list[house_index].label_channel

                elif trans_root == "Level":
                    len_house_list = len(self.house_list)
                    if len_house_list == 0:
                        continue
                    house_index = random.randint(0, len_house_list - 1)

                    len_level_list = len(self.house_list[house_index].level_list)
                    if len_level_list == 0:
                        continue
                    level_index = random.randint(0, len_level_list - 1)

                    if not self.house_list[house_index].level_list[level_index].is_valid:
                        continue

                    return self.house_list[house_index].level_list[level_index].label_channel

                elif trans_root == "Room":
                    len_house_list = len(self.house_list)
                    if len_house_list == 0:
                        continue
                    house_index = random.randint(0, len_house_list - 1)

                    len_level_list = len(self.house_list[house_index].level_list)
                    if len_level_list == 0:
                        continue
                    level_index = random.randint(0, len_level_list - 1)

                    len_room_list = len(self.house_list[house_index].level_list[level_index].room_list)
                    if len_room_list == 0:
                        continue
                    room_index = random.randint(0, len_room_list - 1)

                    if not self.house_list[house_index].level_list[level_index].room_list[room_index].is_valid:
                        continue

                    return self.house_list[house_index].level_list[level_index].room_list[room_index].label_channel

        else:
            house_index, level_index, room_index = room_id_list

            if trans_root == "House":
                if not self.house_list[house_index].is_valid:
                    print("this house is not valid. id :", room_id_list)
                    return None

                return self.house_list[house_index].label_channel

            elif trans_root == "Level":
                level_index = random.randint(0, len(self.house_list[house_index].level_list) - 1)
                if not self.house_list[house_index].level_list[level_index].is_valid:
                    print("this level is not valid. id :", room_id_list)
                    return None

                return self.house_list[house_index].level_list[level_index].label_channel

            elif trans_root == "Room":
                room_index = random.randint(0, len(self.house_list[house_index].level_list[level_index].room_list) - 1)
                if not self.house_list[house_index].level_list[level_index].room_list[room_index].is_valid:
                    print("this room is not valid. id :", room_id_list)
                    return None

                return self.house_list[house_index].level_list[level_index].room_list[room_index].label_channel


class JsonLoader:
    def __init__(self, json_id_list_path=None, csv_path=None, show_bbox=False):
        self.json_id_list_path = None
        self.json_id_list = None
        self.csv = None
        self.json = None
        self.image = None
        self.label_list = None
        self.label_channel = None
        self.show_bbox = show_bbox

        self.max_image_size = [1920, 1080]
        self.scale = 1
        self.trans = [0, 0]

        self.x_index = 0
        self.y_index = 1
        self.z_index = 2

        self.target_image_width = None
        self.target_image_height = None

        self.valid_object_num = 0

        if json_id_list_path is not None:
            self.json_id_list_path = json_id_list_path
            if self.json_id_list_path[-1] != "/":
                self.json_id_list_path += "/"

            self.load_json_id_list(json_id_list_path)

        if csv_path is not None:
            self.load_csv(csv_path)

    def load_json_id_list(self, json_id_list_path):
        self.json_id_list = []

        json_idx = os.listdir(json_id_list_path)

        for id in json_idx:
            self.json_id_list.append(self.json_id_list_path + id + "/house.json")

    def load_csv(self, csv_path):
        self.csv = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)

            for row in reader:
                self.csv.append(row)

        self.label_list = []

        for line in self.csv:
            if line[3] not in self.label_list:
                self.label_list.append(line[3])

    def load_json(self, json_path):
        with open(json_path, "r") as f:
            self.json = json.load(f)

        self.image = None
        self.label_channel = None

        self.scale = 1
        self.trans = [0, 0, 0]

        up = self.json["up"]

        if up[0] == 1:
            self.x_index = 1
            self.y_index = 2
            self.z_index = 0
        elif up[1] == 1:
            self.x_index = 2
            self.y_index = 0
            self.z_index = 1
        else:
            self.x_index = 0
            self.y_index = 1
            self.z_index = 2

        self.valid_object_num = 0

        x_min = self.json["bbox"]["min"][self.x_index]
        x_max = self.json["bbox"]["max"][self.x_index]

        y_min = self.json["bbox"]["min"][self.y_index]
        y_max = self.json["bbox"]["max"][self.y_index]

        z_min = self.json["bbox"]["min"][self.z_index]

        scene_width = x_max - x_min
        scene_height = y_max - y_min

        self.scale = self.max_image_size[0] / scene_width

        new_scale = self.max_image_size[1] / scene_height

        if self.scale > new_scale:
            self.scale = new_scale

        self.target_image_width = int(scene_width * self.scale)
        self.target_image_height = int(scene_height * self.scale)

        self.trans = [- int(x_min * self.scale), - int(y_min * self.scale), - int(z_min * self.scale)]

    def find_bbox(self, max_dist_from_floor=-1):
        if self.show_bbox:
            self.image = np.zeros((self.target_image_height, self.target_image_width, 3))

        self.label_channel = np.zeros((self.target_image_height, self.target_image_width))

        if "levels" in self.json:
            for level in self.json["levels"]:
                if "nodes" in level:
                    for node in level["nodes"]:
                        if "valid" in node and "type" in node:
                            if node["valid"] == 1 and node["type"] != "Room":

                                if "modelId" not in node:
                                    continue

                                bbox = node["bbox"]

                                z_min_from_floor = int(bbox["min"][self.z_index] * self.scale) + self.trans[2]

                                if z_min_from_floor > max_dist_from_floor and max_dist_from_floor != -1:
                                    continue

                                x_min_on_image = int(bbox["min"][self.x_index] * self.scale) + self.trans[0]
                                x_max_on_image = int(bbox["max"][self.x_index] * self.scale) + self.trans[0]

                                y_min_on_image = int(bbox["min"][self.y_index] * self.scale) + self.trans[1]
                                y_max_on_image = int(bbox["max"][self.y_index] * self.scale) + self.trans[1]

                                modelId = node["modelId"]

                                model_class = ""

                                for line in self.csv:
                                    if line[1] == modelId:
                                        model_class = line[3]
                                        break

                                if model_class == "":
                                    continue

                                current_label_index = 0

                                for i in range(len(self.label_list)):
                                    if model_class == self.label_list[i]:
                                        current_label_index = i + 1
                                        break

                                if current_label_index > 0:
                                    self.valid_object_num += 1
                                    cv2.rectangle(self.label_channel, (x_min_on_image, y_min_on_image), (x_max_on_image, y_max_on_image), current_label_index, -1)

                                if self.show_bbox:
                                    cv2.rectangle(self.image, (x_min_on_image, y_min_on_image), (x_max_on_image, y_max_on_image), (current_label_index / len(self.label_list), current_label_index / len(self.label_list), current_label_index / len(self.label_list)), -1)

    def create_label_channel(self, index=-1):
        if index < 0 or index >= len(self.json_id_list):
            index = random.randint(0, len(self.json_id_list) - 1)

        self.load_json(self.json_id_list[index])

        current_max_dist_to_floor = 1

        self.find_bbox()

        all_object_num = self.valid_object_num

        self.valid_object_num = 0

        while self.valid_object_num / all_object_num < 0.5:
            self.find_bbox(current_max_dist_to_floor)

            current_max_dist_to_floor += 2

        if self.show_bbox:
            cv2.imshow("bbox", self.image)
            cv2.imshow("label", self.label_channel)
            cv2.waitKey(0)

        return self.label_channel

if __name__ == "__main__":

    root_path = "D:/chLi/Download/installer/SUNCG/suncg_data/suncg_data/"

    dataset_path = root_path + "house/"
    target_width = 32
    target_height = 32
    size_fixed = True
    csv_path = root_path + "SUNCGtoolbox-master/metadata/ModelCategoryMapping.csv"
    valid_label_list = ["desk", "chair", "table", "door", "window", "sofa", "bed", "curtain", "shelving"]
    is_binary = False
    min_node_num = 5
    save_object = ["Room"]
    save_path = "D:/chLi/Download/installer/SUNCG/label_channel.npy"
    channel_num = 1
    use_color = False
    free_label = True
    number_type = np.uint8

    #### method : load from .json files and compute label channel each time
    #### fps:30
    # json_loader = JsonLoader(dataset_path, csv_path, True)

    # current_cycle_times = 0
    # start_time = time()
    # while True:
    #     json_loader.create_label_channel()
    #     current_cycle_times += 1
    #     print("\r", current_cycle_times, "fps =", int(1.0 * current_cycle_times / (time() - start_time)), "    ", end="")

    # print()

    #### method : first load all .json files, then return the label channel needed directly
    #### fps:10000
    suncg = SUNCGDataBase(dataset_path,
                            target_width,
                            target_height,
                            size_fixed,
                            csv_path,
                            valid_label_list,
                            is_binary,
                            min_node_num,
                            save_object,
                            save_path,
                            channel_num,
                            use_color,
                            free_label,
                            number_type
                        )

    exit()

    current_cycle_times = 0
    start_time = time()
    while True:
        label_channel = suncg.load_label_channel("Room")
        ### if use this, Ctrl+F:"cv2.rectangle" and let all "label_index" become "1.0 * label_index / len(label.valid_label_list)" ###
        # cv2.imshow("label channel", label_channel)
        # cv2.waitKey(0)
        current_cycle_times += 1
        if time() > start_time:
            print("\r", current_cycle_times, "fps =", int(1.0 * current_cycle_times / (time() - start_time)), "    ", end="")

    print()
