from typing import *
import itertools
import numpy as np

def get_shape(obj: Any) -> Tuple[int]:
    return np.array(obj).shape

def shape_after_indexing(original_shape: Tuple[int], indexing_obj: Any) -> Tuple[int]:
    # If you're looking at A[0], type(indexing_obj) == int, but for A[0, 1], type(indexing_obj) ==
    # tuple. So we need to homogenize the input.
    if isinstance(indexing_obj, tuple):
        indexing_list = list(indexing_obj)
    else:
        indexing_list = [indexing_obj]

    # This is to do with indexing by lists — advanced indexing. We collapse adjacent lists into a
    # single object called a ListBlob.
    # e.g. [[0, 0], [1, 0], 1:2, [2, 0]] --> [<ListBlob ...>, 1:2, <ListBlob ...>]
    with_list_blobs = collapse_lists_to_blobs(indexing_list)
    list_blobs = [item for _, item in with_list_blobs if isinstance(item, ListBlob)]
    many_list_blobs = len(list_blobs) > 1

    result = []
    if many_list_blobs:
        # If many list blobs, we put the result on the front of the result.
        # Here you'd assert they all have the same shape or can be broadcast into the same shape.
        result = [*list_blobs[0].shape]

    for axis, item in with_list_blobs:
        if isinstance(item, int):
            # Do nothing — the original value will not be included in the result.
            pass
        elif isinstance(item, slice):
            original_value = original_shape[axis]
            if item == slice(None): # slice(None) is also known as ':'
                new_value = original_value
                result.append(new_value)
            else:
                start = item.start or 0
                stop = item.stop or original_value
                stop = min(stop, original_value) # it should not exceed the length of this axis.
                step = item.step or 1

                new_value = (stop - start) // step
                result.append(new_value)
        elif isinstance(item, ListBlob):
            if not many_list_blobs:
                # This is the only list blob — we replace the original value where it occurred with
                # the new shape.
                result.extend(item.shape)
                axis = item.end # necessary in case this is the last object in the list.
        elif item is None:
            # This is np.newaxis.
            result.append(1)
        else:
            print(item)
            raise ValueError

    # Fill in the remaining dimensions as though the rest of the indexing tuple were filled in
    # with ':'.
    result.extend(original_shape[axis+1:])

    return tuple(result)

class ListBlob:
    def __init__(self, shape):
        self.shape = shape
        self.end = None

    def __repr__(self):
        return f"<ListBlob: shape={self.shape}, end={self.end}>"

def collapse_lists_to_blobs(indexing_list: List[Any]) -> List[Any]:
    result = []
    for is_list, group in itertools.groupby(enumerate(indexing_list), key=lambda t: isinstance(t[1], list)):
        contents = list(group)
        # Here you'd assert they all have the same shape or can be broadcast into the same shape.
        first_axis, first_item = contents[0]
        
        if is_list:
            result.append((first_axis, ListBlob(get_shape(first_item))))
        else:
            result.extend(contents)
    
    # Make each blob aware of when it ends.
    for (_, a_item), (b_axis, _) in zip(result, result[1:]):
        if isinstance(a_item, ListBlob):
            a_item.end = b_axis
    _, last_item = result[-1] # The last one needs special-casing.
    if isinstance(last_item, ListBlob):
        last_item.end = len(indexing_list) - 1

    return result
