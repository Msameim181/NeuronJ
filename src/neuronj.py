import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rich.progress import track


def NeuronJ(data_addr:str, 
            output_dir:str,
            image_dir_name:str = 'images', 
            mask_dir_name:str = 'masks', 
            image_ext:str = '.tif', 
            mask_ext:str = '.tif',
            colorize:bool = True):

    """NeuronJ data pre-processing.

    Args:
        data_addr (str): Address of data folder. In this folder should be pairs of images and masks.
        output_dir (str): Address of output folder.
        image_dir_name (str): Name of image folder.
        mask_dir_name (str): Name of mask folder.
        image_ext (str): Extension of image file.
        mask_ext (str): Extension of mask file.
        colorize (bool): If True, colorize the image.

    Returns:
        None
    """
    neuron_colorType = {
        '0': '#adadad',  # 'Secondary', Grey
        '1': '#d52c2c',  # 'Axon',      Red
        '2': '#1985bf',  # 'Dendrite',  Blue
        '3': '#d3d022',  # 'Primary',   Yellow
        '4': '#adadad',  # 'Secondary', Grey
        '6': '#25a214',  # 'Dendrite_primary', Green
    }
    ndfs_addr, images_addr = Data_finder(data_addr)

    file_name, output_addresses = Output_checker(output_dir, image_dir_name, mask_dir_name)
    
    for ndf_addr, image_addr in track(zip(ndfs_addr, images_addr), total=len(ndfs_addr)):
        lines, img = Data_loader(ndf_addr, image_addr)
        trace_sections = Trace_finder(lines)
        trace_data = Trace_data_extractor(trace_sections, neuron_colorType)
        Save_result(image_addr, trace_data, file_name, output_addresses, 
                    image_ext, mask_ext, colorize)

        file_name += 1

def Data_finder(data_addr:str):
    """Loads data from a given address. '.ndf' & '.tif' files are expected to be in the folder.

    Args:
        data_addr (str): Address of data folder. In this folder should be pairs of images and masks.
        images format: 'image_name.tif'
        masks format: 'mask_name.ndf'
        And these name should be the same for each sample.

    Returns:
        tuple: contains:
            ndfs_addr: A list of '.ndf' files addresses.
            images_addr: A list of '.tif' files addresses.
    """
    data_addr = Path(data_addr)
    ndfs_addr = [os.path.join(data_addr, item) for item in os.listdir(data_addr) if item.endswith('.ndf')]
    images_addr = [os.path.join(data_addr, item) for item in os.listdir(data_addr) if item.endswith('.tif')]
    ndfs_addr.sort(), images_addr.sort()
    return ndfs_addr, images_addr

def Data_loader(ndf_addr:str, image_addr:str):
    """Load data from file to variable.

    Args:
        ndf_addr (str): Address of one '.ndf' file.
        image_addr (str): Address of one '.tif' file.
        

    Returns:
        tuple: contains:
            lines: A list of lines from '.ndf' file.
            img: A numpy array of image. (with plot)
    """
    with open(ndf_addr) as file:
        lines = file.readlines()
        file.close()
    
    img = plt.imread(image_addr)

    return lines, img

def Trace_finder(lines:list):
    """Finds traces index from data. And extract eact trace section from data

    Args:
        lines (list): A list of lines from '.ndf' file.

    Returns:
        trace_sections (list): A list of trace sections.
    """
    trace_index = []
    # segment_index = []
    for line in lines:
        if line.startswith('// Tracing'):
            trace_index.append(lines.index(line))
        # if line.startswith('// Segment'):
        #     segment_index.append(lines.index(line))

    trace_sections = []
    for t in trace_index:
        if trace_index.index(t) < len(trace_index)-1:
            trace_sections.append(lines[t:trace_index[trace_index.index(t)+1]])
        else:
            trace_sections.append(lines[t:])
    
    return trace_sections

def Segment_finder(trace:list):
    """Finds Segment index from data. And extract eact segment section from data then save into a list.

    Args:
        trace (list): A list of lines from each trace.

    Returns:
        segments_data (list): A list of segments data.
        segments_names (list): A list of segments names.
    """
    segments_index = []
    for line in trace:
        if line.startswith('// Segment'):
            segments_index.append(trace.index(line))

    segments_data = []
    for t in segments_index:
        if segments_index.index(t) < len(segments_index)-1:
            segments_data.append(trace[t:segments_index[segments_index.index(t)+1]])
        else:
            segments_data.append(trace[t:])

    segments_names = [trace[item].replace('\n', '').replace('// ', '') for item in segments_index]
    
    return segments_data, segments_names

def Trace_data_extractor(trace_sections:list, neurite:dict):
    """Extract trace data into a list of dict of information.

    Dictionary keys:
    'name': name of trace
    'val_one': value of first point (Number of Trace)
    'val_two': value of second point (Type of neuron; neurite)
    'val_three': value of third point (Default: 0)
    'mood': mood of trace (Default)
    'segment_data': list of segment data (in each segment (x,y) coordinates as pairs)
    'segment_data_XY': list of segment data (in each segment (x) and (y) coordinates are separated)
    'segments_names': list of segment names (for each segment)
    'length': length of trace (length of segments)
    'color': color of trace (Default: '#d52c2c')

    Args:
        trace_sections (list): A list of trace sections.

    Returns:
        trace_data (list): A list of lists. Each list contains trace data.
    """
    trace_data = []
    for trace in trace_sections:

        name = trace.pop(0).replace('\n', '').replace('// ', '')
        val_one = int(trace.pop(0).replace('\n', ''))
        val_two = int(trace.pop(0).replace('\n', ''))
        val_three = int(trace.pop(0).replace('\n', ''))
        mood = trace.pop(0).replace('\n', '')
        color = neurite[str(val_two)]

        segmentaion_data, segments_names = Segment_finder(trace)

        segment_data = []
        segment_data_XY = []
        for each_segment in segmentaion_data:
            datas = each_segment[1:]
            if datas[-1].startswith('//'):
                datas.pop(-1)
            datas = [int(item.replace('\n', '')) for item in datas]
            x = datas[0::2]
            y = datas[1::2]
            realdata = list(zip(x, y))
            segment_data.append(realdata)
            segment_data_XY.append((x, y))

        dict = {'name': name, 'val_one': val_one, 'val_two': val_two, 'val_three': val_three, 'mood': mood,
            'segment_data': segment_data, 'segment_data_XY': segment_data_XY, 'segments_names': segments_names,
            'length': len(segments_names), 'color': color}
        
        trace_data.append(dict)
    return trace_data

def Output_checker(output_addr:str,
                image_dir_name:str = 'images', 
                mask_dir_name:str = 'masks'):
    """Check if output folder exists. If not, create one.

    Returns:
        output_addr: Address of output folder.
    """
    output_addr = Path(output_addr)
    # if not output_addr.exists():
    #     os.mkdir(output_addr)
    output_addr_list = output_addr.parts
    temp_path = ""
    for item in output_addr_list:
        temp_path = Path(temp_path, item)
        if not temp_path.exists():
            os.mkdir(temp_path)

    image_output_addr = Path(output_addr, image_dir_name)
    if not image_output_addr.exists():
        os.mkdir(image_output_addr)
    
    mask_output_addr = Path(output_addr, mask_dir_name)
    if not mask_output_addr.exists():
        os.mkdir(mask_output_addr)

    show_output_addr = Path(output_addr, 'image_mask')
    if not show_output_addr.exists():
        os.mkdir(show_output_addr)   

    images_addr = [os.path.join(image_output_addr, item) for item in os.listdir(image_output_addr) if item.endswith('.tif')]
    # masks_addr = [os.path.join(mask_output_addr, item) for item in os.listdir(mask_output_addr) if item.endswith('.tif')]
    # show_addr = [os.path.join(show_output_addr, item) for item in os.listdir(show_output_addr) if item.endswith('.tif')]
    images_addr.sort()
    # masks_addr.sort(), show_addr.sort()

    if len(images_addr):
        file_name = int(images_addr[-1].split('\\')[-1].split('.')[0]) + 1 
    else:
        file_name = 1
    addresses = [output_addr, image_output_addr, mask_output_addr, show_output_addr]
    return file_name, addresses

def file_name_fixer(file_name:int, file_ext:str = '.tif'):
    """Fix file name if it is not in the correct format.

    Args:
        file_name (int): File name in integer.
        file_ext (str): File extension.

    Returns:
        file_name (str): Fixed file name.
    """
    if file_name < 10:
        file_name = '000' + str(file_name)
    elif file_name < 100:
        file_name = '00' + str(file_name)
    elif file_name < 1000:
        file_name = '0' + str(file_name)
    else:
        file_name = str(file_name)

    if file_name.endswith(file_ext):
        pass
    else:
        file_name = file_name + file_ext

    return file_name

def Save_result(image_addr:str,
                trace_data:list, 
                file_name:int,
                output_addresses:list, 
                image_ext:str = '.tif', 
                mask_ext:str = '.tif',
                colorize:bool = True):
    
    """Save result into output folder.

    Args:
        image_addr (str): Address of image.
        trace_data (list): A list of lists. Each list contains trace data.
        file_name (int): File name in integer.
        output_addresses (list): A list of addresses.
        image_ext (str): Image extension.
        mask_ext (str): Mask extension.
        colorize (bool): If True, colorize the image.
    
    Returns:
        None
    """
    img_file_name = file_name_fixer(file_name, image_ext)
    output_addr, image_output_addr, mask_output_addr, show_output_addr = output_addresses
    
    # Copy original image
    destination_addr = shutil.copy(image_addr, image_output_addr)
    os.rename(destination_addr, Path(image_output_addr, img_file_name))

    # Create mask
    img = plt.imread(image_addr)
    image_size = (img.shape[0], img.shape[1])
    black_backgorund = np.zeros(image_size, dtype=np.uint8)
    plt.figure(figsize = (5, 5), dpi = 300)
    for trace in trace_data:
        if colorize:
            trace_color = trace['color']
        else:
            trace_color = 'white'
        for segment in trace['segment_data_XY']:
            x = segment[0]
            y = segment[1]
            plt.plot(x, y, color=trace_color, markersize=0.3, linewidth=0.3)
    plt.axis('off')
    plt.imshow(black_backgorund, cmap='gray')
   
    # Save mask
    mask_file_name = file_name_fixer(file_name, mask_ext)
    mask_save_addr = Path(mask_output_addr, mask_file_name)
    plt.savefig(mask_save_addr, dpi = 330.33, bbox_inches='tight', pad_inches = 0)
    plt.close()


    # Create show image
    plt.figure(figsize = (5, 5), dpi = 150)
    for trace in trace_data:
        if colorize:
            trace_color = trace['color']
        else:
            trace_color = 'white'
        for segment in trace['segment_data_XY']:
            x = segment[0]
            y = segment[1]
            plt.plot(x, y, color=trace_color, markersize=0.3, linewidth=0.3)
    plt.axis('off')
    plt.imshow(img, cmap='gray')

    # Save Show image
    show_file_name = file_name_fixer(file_name, '.tif')
    show_save_addr = Path(show_output_addr, show_file_name)
    plt.savefig(show_save_addr, dpi = 330.33, bbox_inches='tight', pad_inches = 0)
    plt.close()
    plt.rcParams.update({'figure.max_open_warning': 100})
