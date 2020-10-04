# coding=utf-8
import os
import shutil
import zipfile
from collections import defaultdict


class NER(object):
    def __init__(self, tid, start, end, ttype, text=''):
        self.tid = str(tid).strip()
        self.start = int(start)
        self.end = int(end)
        self.text = str(text).strip()
        self.ttype = str(ttype).strip()

    def span_matches(self, other, mode='strict'):
        assert mode in ('strict', 'lenient')
        if mode == 'strict':
            if self.start == other.start and self.end == other.end:
                return True
        else:
            if (self.end > other.start and self.start < other.end) or \
                    (self.start < other.end and other.start < self.end):
                return True
        return False

    def equals(self, other, mode='strict'):
        assert mode in ('strict', 'lenient')
        return other.ttype == self.ttype and self.span_matches(other, mode)

    def __str__(self):
        return '{}\t{}\t({}:{})'.format(self.ttype, self.text, self.start, self.end)


class RecordTrack(object):

    def __init__(self, file_path):
        self.path = os.path.abspath(file_path)
        self.basename = os.path.basename(self.path)
        self.annotations = self._get_annotations()

    @property
    def tags(self):
        return self.annotations['tags']

    def _get_annotations(self):
        annotations = defaultdict(dict)
        with open(self.path) as annotation_file:
            lines = annotation_file.readlines()
            for line_num, line in enumerate(lines):
                if line.strip().startswith('T'):
                    try:
                        tag_id, tag_m, tag_text = line.strip().split('\t')
                    except ValueError:
                        print(self.path, line)
                    # adapt to Brat tool:
                    if len(tag_m.split(' ')) == 3:
                        tag_type, tag_start, tag_end = tag_m.split(' ')
                    elif len(tag_m.split(' ')) == 4:
                        tag_type, tag_start, _, tag_end = tag_m.split(' ')
                    elif len(tag_m.split(' ')) == 5:
                        tag_type, tag_start, _, _, tag_end = tag_m.split(' ')
                    else:
                        print(self.path)
                        print(line)
                    tag_start, tag_end = int(tag_start), int(tag_end)
                    annotations['tags'][tag_id] = NER(tag_id, tag_start, tag_end, tag_type, tag_text)

        return annotations


def parse_ann_file(ann_file):
    return RecordTrack(ann_file)


'''
    解压 压缩文件 到 解压目录, 返回解压后的答案目录，要求答案目录下存放生成的ann文件
    @@ extract_dir: 解压目录
    @@ zip_file: 选手上传的zip文件
'''


def get_answer_dir(extract_dir, zip_file):
    answer_dir = ''

    if os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir)

    with zipfile.ZipFile(zip_file, "r") as zip_data:
        zip_data.extractall(extract_dir)
        zip_data.close()

    # 遍历解压后的目录，取首次出现的目录，如果存在一些不相关目录，如MACOSX之类的，请删除掉
    for item in os.listdir(extract_dir):
        answer_dir = '/'.join([extract_dir, item])
        if os.path.isdir(answer_dir):
            break

    return answer_dir


if __name__ == "__main__":
    '''
      format checker
    '''

    # NOTE: 实际测试时候请替换成选手自己机器的目录
    extract_dir = '/Users/tianchi/Downloads/tmp'
    zip_file = '/Users/tianchi/Downloads/train.zip'

    ## check zip file, 要求解压后得到的answer_dir目录存放生成的ann文件
    answer_dir = get_answer_dir(extract_dir, zip_file)
    print('Answer dir: ', answer_dir)

    ## check ann file format:
    ann_file = '/Users/tianchi/Downloads/tmp/train/1.ann'
    record = parse_ann_file(ann_file)
    print('Total annotation number: ', len(record.tags))
