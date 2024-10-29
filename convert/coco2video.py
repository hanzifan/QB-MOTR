import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser("image to video converter", add_help=False)
    parser.add_argument("--src_json", default="/home/hzf/data/Refer-Youtube_VIS/train/train.json", type=str)
    parser.add_argument("--det", default="/home/hzf/data/Refer-Youtube_VIS/train/labels", type=str)
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Convert starting', parents=[parse_args()])
    args = parser.parse_args()
    src_anno = json.load(open(args.src_json, 'r'))['annotations'] 
    src_videos = json.load(open(args.src_json, 'r'))['videos'] 
    os.makedirs(args.det, exist_ok=True)

    # set seq framework
    dict_seq = {}
    for video in src_videos:
        dict_frame = []
        dict_seq[video['id']] = dict_frame
    # set every instance annotation
    for anno in src_anno:
        if len(dict_seq[anno['video_id']]) == 0:
            for idx in range(len(anno['bboxes'])):
                dict_instance = {}
                dict_instance['name'] = src_videos[anno['video_id']-1]['file_names'][idx]
                dict_instance['labels'] = []
                dict_labels = {}
                dict_labels['id'] = anno['id']
                # dict_labels['category'] = anno['category_id']
                dict_labels['category'] = 'pedestrian'
                dict_attributes = {}
                dict_attributes['occluded'] = False
                dict_attributes['truncated'] = False
                dict_attributes['crowd'] = False
                dict_labels['attributes'] = dict_attributes
                if anno['bboxes'][idx] != None:
                    dict_bboxes = {}
                    dict_bboxes['x1'] = anno['bboxes'][idx][0]
                    dict_bboxes['y1'] = anno['bboxes'][idx][1]
                    dict_bboxes['x2'] = anno['bboxes'][idx][0] + anno['bboxes'][idx][2]
                    dict_bboxes['y2'] = anno['bboxes'][idx][1] + anno['bboxes'][idx][3]
                    dict_labels['box2d'] = dict_bboxes
                else:
                    dict_bboxes = {}
                    dict_bboxes['x1'] = 0.0
                    dict_bboxes['y1'] = 0.0
                    dict_bboxes['x2'] = 0.0
                    dict_bboxes['y2'] = 0.0
                    dict_labels['box2d'] = dict_bboxes
                dict_instance['labels'].append(dict_labels)
                dict_instance['videoName'] = src_videos[anno['video_id']-1]['file_names'][idx][:-10]
                dict_instance['frameIndex'] = idx
                dict_seq[anno['video_id']].append(dict_instance)
        else:
            for idx in range(len(anno['bboxes'])):
                dict_labels = {}
                dict_labels['id'] = anno['id']
                # dict_labels['category'] = anno['category_id']
                dict_labels['category'] = 'pedestrian'
                dict_attributes = {}
                dict_attributes['occluded'] = False
                dict_attributes['truncated'] = False
                dict_attributes['crowd'] = False
                dict_labels['attributes'] = dict_attributes
                if anno['bboxes'][idx] != None:
                    dict_bboxes = {}
                    dict_bboxes['x1'] = anno['bboxes'][idx][0]
                    dict_bboxes['y1'] = anno['bboxes'][idx][1]
                    dict_bboxes['x2'] = anno['bboxes'][idx][0] + anno['bboxes'][idx][2]
                    dict_bboxes['y2'] = anno['bboxes'][idx][1] + anno['bboxes'][idx][3]
                    dict_labels['box2d'] = dict_bboxes
                else:
                    dict_bboxes = {}
                    dict_bboxes['x1'] = 0.0
                    dict_bboxes['y1'] = 0.0
                    dict_bboxes['x2'] = 0.0
                    dict_bboxes['y2'] = 0.0
                    dict_labels['box2d'] = dict_bboxes
                dict_seq[anno['video_id']][idx]['labels'].append(dict_labels)

    # dump the json file
    for idx in range(len(dict_seq)):
        save_path = '/home/hzf/data/Refer-Youtube_VIS/train/labels/'
        save_path = save_path + dict_seq[idx+1][0]['videoName'] + '.json'
        with open(save_path,'w') as file:
            json.dump(dict_seq[idx+1],file)
        