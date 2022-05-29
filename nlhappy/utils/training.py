from typing import List

def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'S':
        chunk_end = True
    # pred_label中可能出现这种情形
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start

def convert_tags_to_spans(seq: List[str]):
    """
    将bio或bios标注的序列转换为span形式(label, start, end).
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        convert_tags_to_spans(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def align_char_span(char_span_offset: List, token_offset_mapping, special_offset=(0,0)) -> List:
    '''对齐字符级别的span标签与bert的子词切分的标签
    参数:
    - char_span_offset: 字符级别的文本片段下标,例如(0,1)
    - token_offset_mapping: 词符与字符下标映射例如[(0,1), (1,2), (2,3) ...]
    输出
    token_span_offset: 词符级别的文本片段下标,例如(0,2)
    '''
    token_span_offset = []
    for i, offset in enumerate(token_offset_mapping):
        if offset != special_offset:
            if offset[0] == char_span_offset[0]:
                start = i
                if offset[1] == char_span_offset[1]:
                    end = i+1
                    break
                else: next
            elif offset[1] == char_span_offset[1]:
                end = i+1
                break
    try:
        if start and end:
            token_span_offset = [start, end]
    except Exception:
        pass
    return token_span_offset


def align_token_span(token_span_offset, token_offset_mapping):
    '''将词符级别的下标对齐为字符级别的下标
    参数
    - token_span_offset: 例如(0, 1) 下标指的是字符的下标
    - token_offset_mapping: 每个词符与字符对应的下标[(0,1),(1,2)]
    返回
    char_span_offset: (0,2)
    '''
    char_span_offset = ()
    if token_span_offset[1] - token_span_offset[0] == 1:
        char_span_offset = token_offset_mapping[token_span_offset[0]]
        return char_span_offset

    else:
        start = token_offset_mapping[token_span_offset[0]][0]
        end = token_offset_mapping[token_span_offset[1]-1][1]
        char_span_offset = (start, end)
        return char_span_offset
