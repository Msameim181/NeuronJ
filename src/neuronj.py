import os
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)

progress = Progress(
    SpinnerColumn(finished_text="[bold blue]:heavy_check_mark:", style='blue'),
    TextColumn("[bold blue]{task.description}", justify="right"),
    BarColumn(bar_width=50),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    TimeRemainingColumn(),
    "•",
    TimeElapsedColumn(),
    "•",
    "[progress.filesize]Passed: {task.completed} item",
    "•",
    "[progress.filesize.total]Total: {task.total} item",
)

def NeuronJ(data_addr:str, 
            output_dir:str,
            image_dir_name:str = 'images', 
            mask_dir_name:str = 'masks', 
            image_ext:str = '.tif', 
            mask_ext:str = '.tif',
            colorize:bool = False,
            save_data_dpi:int = 300,
            resize_mask_to_image_size:bool = False,
            resize_lib:str = 'pillow',
            mask_builder:str = 'pillow'):

    """NeuronJ data pre-processing.

    Args:
        data_addr (str): Address of data folder. In this folder should be pairs of images and masks.
        output_dir (str): Address of output folder.
        image_dir_name (str): Name of image folder.
        mask_dir_name (str): Name of mask folder.
        image_ext (str): Extension of image file.
        mask_ext (str): Extension of mask file.
        colorize (bool): If True, colorize the image.
        save_data_dpi (int): DPI of saved data. Default: 300.
        resize_mask_to_image_size (bool): If True, resize mask to image size.
        resize_lib (str): Library used for resizing. Default: pillow. [pillow, opencv].
        mask_builder (str): Library used for mask building. Default: matplotlib. [matplotlib, pillow, opencv]. 

        ** Note: matplotlib is recommended for quality. Pillow and opencv are faster but not recommended. Pillow has no image_mask output. OpenCV not support colorize.
        Quality: matplotlib > pillow > opencv
        Speed: pillow > opencv > matplotlib
        Output File size (image_masks: presentaion output): matplotlib > opencv
        Output File size (masks): matplotlib (no resizeing) > matplotlib (pillow resizing) > pillow > matplotlib (opencv resizing) > opencv
        Compression: matplotlib (pillow resizing) and pillow no compression. matplotlib (opencv resizing) and opencv have LZW compression.
        DPI: Can be modified in just in matplotlib with no resize and compression. With resize or other methods, DPI is 96. 

        ** Note: If you used pillow or opencv as mask_builder, there is no need to resize mask to image size. 

    Returns:
        None
    """
    neuron_colorType = {
        '0': {'hex': '#adadad', 'label': 'Secondary',        'name': 'grey'  , 'rgb': (173, 173, 173)},
        '1': {'hex': '#d52c2c', 'label': 'Axon',             'name': 'red'   , 'rgb': (213, 44, 44)},
        '2': {'hex': '#1985bf', 'label': 'Dendrite',         'name': 'blue'  , 'rgb': (25, 133, 191)},
        '3': {'hex': '#d3d022', 'label': 'Primary',          'name': 'yellow', 'rgb': (211, 208, 34)},
        '4': {'hex': '#adadad', 'label': 'Secondary',        'name': 'grey'  , 'rgb': (173, 173, 173)},
        '6': {'hex': '#25a214', 'label': 'Dendrite_primary', 'name': 'green' , 'rgb': (37, 162, 20)},
    }
    ndfs_addr, images_addr = Data_finder(data_addr)

    file_name, output_addresses = Output_checker(output_dir, image_dir_name, mask_dir_name)
    with progress:
        for ndf_addr, image_addr in progress.track(zip(ndfs_addr, images_addr), total=len(ndfs_addr), 
                                                    description='Working on data...'):
            lines, img = Data_loader(ndf_addr, image_addr)
            trace_sections = Trace_finder(lines)
            trace_data = Trace_data_extractor(trace_sections, neuron_colorType)
            Save_result(image_addr, trace_data, file_name, output_addresses, 
                        image_ext, mask_ext, colorize, save_data_dpi,
                        resize_mask_to_image_size, resize_lib, mask_builder)

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

    temp_addr = Path(output_addr, 'temp')
    if not temp_addr.exists():
        os.mkdir(temp_addr)   

    images_addr = [os.path.join(image_output_addr, item) for item in os.listdir(image_output_addr) if item.endswith('.tif')]
    # masks_addr = [os.path.join(mask_output_addr, item) for item in os.listdir(mask_output_addr) if item.endswith('.tif')]
    # show_addr = [os.path.join(show_output_addr, item) for item in os.listdir(show_output_addr) if item.endswith('.tif')]
    images_addr.sort()
    # masks_addr.sort(), show_addr.sort()

    if len(images_addr):
        file_name = int(images_addr[-1].split('\\')[-1].split('.')[0]) + 1 
    else:
        file_name = 1
    addresses = [output_addr, image_output_addr, mask_output_addr, show_output_addr, temp_addr]
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

def Image_resizer(image_size:np.array, mask_save_addr:str, lib:str='pillow'):
    """Resize mask to the same size of image.

    Args:
        image_size (np.array): Size of image.
        mask_save_addr (str): Address of mask.
        lib (str): Library to use. [pillow, opencv]
    
    Returns:
        None
    """

    if lib == 'pillow':
        Image.open(mask_save_addr).resize(image_size).save(mask_save_addr)

    elif lib == 'opencv':
        mask_save_addr = str(Path(mask_save_addr))
        img = cv2.imread(mask_save_addr)
        img = cv2.resize(img, image_size)
        cv2.imwrite(mask_save_addr, img)
        ...

def matplotlib_mask_builder(image_addr:str,  
                            trace_data:list, 
                            file_name:int, 
                            mask_output_addr:str, 
                            show_output_addr:str,
                            mask_ext:str = '.tif',
                            colorize:bool = True,
                            save_data_dpi:int = 300,
                            resize_mask_to_image_size:bool = False,
                            resize_lib:str = 'pillow'):
    """Build mask for each trace.

    Args:
        image_addr (str): Address of image.
        trace_data (list): Trace data.
        file_name (int): File name.
        mask_output_addr (str): Address of mask output folder.
        show_output_addr (str): Address of show output folder.
        mask_ext (str): Extension of mask.
        colorize (bool): Colorize mask.
        save_data_dpi (int): DPI of saved data.
        resize_mask_to_image_size (bool): Resize mask to image size.
        resize_lib (str): Library to use. [pillow, opencv]

    Returns:
        None

    """
    
    # Create mask
    img = plt.imread(image_addr)
    image_size = (img.shape[0], img.shape[1])
    black_backgorund = np.zeros(image_size, dtype=np.uint8)
    plt.figure(figsize = (5, 5), dpi = 300)
    for trace in trace_data:
        if colorize:
            trace_color = trace['color']['hex']
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
    plt.savefig(mask_save_addr, dpi = save_data_dpi, bbox_inches='tight', pad_inches = 0)
    plt.close()
    if resize_mask_to_image_size:
        image_size = (image_size[1], image_size[0])
        Image_resizer(image_size, mask_save_addr, resize_lib)


    # Create show image
    plt.figure(figsize = (5, 5), dpi = 150)
    for trace in trace_data:
        if colorize:
            trace_color = trace['color']['hex']
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
    plt.savefig(show_save_addr, dpi = save_data_dpi, bbox_inches='tight', pad_inches = 0)
    plt.close()
    plt.rcParams.update({'figure.max_open_warning': 100})

def pillow_mask_builder(image_addr:str,  
                        trace_data:list, 
                        file_name:int, 
                        mask_output_addr:str, 
                        show_output_addr:str,
                        mask_ext:str = '.tif',
                        colorize:bool = True):
    """Build mask for each trace.

    Args:
        image_addr (str): Address of image.
        trace_data (list): Trace data.
        file_name (int): File name.
        mask_output_addr (str): Address of mask output folder.
        show_output_addr (str): Address of show output folder.
        mask_ext (str): Extension of mask.
        colorize (bool): Colorize mask.

    Returns:
        None

    """

    # Create mask
    img = Image.open(image_addr)
    image_size = img.size
    black_backgorund = Image.new('RGB', image_size, (0, 0, 0))
    drawline = ImageDraw.Draw(black_backgorund)
    for trace in trace_data:
        if colorize:
            trace_color = trace['color']['hex']
        else:
            trace_color = 'white'
        for segment in trace['segment_data']:
            drawline.line(segment, fill=trace_color, width=0)
    # black_backgorund.show()
   
    # Save mask
    mask_file_name = file_name_fixer(file_name, mask_ext)
    mask_save_addr = Path(mask_output_addr, mask_file_name)
    black_backgorund.save(mask_save_addr)
    black_backgorund.close()

    # ------------------ Need Review ------------------
    # Create show image
    # drawline = ImageDraw.Draw(img)
    # for trace in trace_data:
    #     if colorize:
    #         trace_color = trace['color']['hex']
    #     else:
    #         trace_color = 'white'
    #     for segment in trace['segment_data']:
    #         drawline.line(segment, fill=trace_color, width=0)
    # # img.show()

    # # Save Show image
    # show_file_name = file_name_fixer(file_name, '.tif')
    # show_save_addr = Path(show_output_addr, show_file_name)
    # img.save(show_save_addr)
    # img.close()
    # ------------------ Need Review ------------------
    ...

def opencv_mask_builder(image_addr:str,  
                        trace_data:list, 
                        file_name:int, 
                        mask_output_addr:str, 
                        show_output_addr:str,
                        mask_ext:str = '.tif',
                        colorize:bool = True):
    """Build mask for each trace.

    Args:
        image_addr (str): Address of image.
        trace_data (list): Trace data.
        file_name (int): File name.
        mask_output_addr (str): Address of mask output folder.
        show_output_addr (str): Address of show output folder.
        mask_ext (str): Extension of mask.
        colorize (bool): Colorize mask.

    Returns:
        None

    """

    # Create mask

    img = cv2.imread(image_addr)
    image_size = (img.shape[0], img.shape[1])
    black_backgorund = np.zeros(image_size, dtype=np.uint8)
    for trace in trace_data:
        if colorize:
            trace_color = trace['color']['rgb']
        else:
            trace_color = (255, 255, 255)
        for segment in trace['segment_data']:
            for i in range(len(segment)):
                if i < len(segment)-1:
                    cv2.line(black_backgorund, segment[i], segment[i+1], trace_color, thickness=1)
    # cv2.imshow(f"{file_name}", black_backgorund)
    # cv2.waitKey()
   
    # Save mask
    mask_file_name = file_name_fixer(file_name, mask_ext)
    mask_save_addr = Path(mask_output_addr, mask_file_name)
    cv2.imwrite(str(mask_save_addr), black_backgorund)


    # Create show image
    for trace in trace_data:
        if colorize:
            trace_color = trace['color']['rgb']
        else:
            trace_color = (255, 255, 255)
        for segment in trace['segment_data']:
            for i in range(len(segment)):
                if i < len(segment)-1:
                    cv2.line(img, segment[i], segment[i+1], trace_color, thickness=1)
    # cv2.imshow(f"{file_name}", img)
    # cv2.waitKey()

    # Save Show image
    show_file_name = file_name_fixer(file_name, '.tif')
    show_save_addr = Path(show_output_addr, show_file_name)
    cv2.imwrite(str(show_save_addr), img)

def Save_result(image_addr:str,
                trace_data:list, 
                file_name:int,
                output_addresses:list, 
                image_ext:str = '.tif', 
                mask_ext:str = '.tif',
                colorize:bool = True,
                save_data_dpi:int = 300,
                resize_mask_to_image_size:bool = False,
                resize_lib:str = 'pillow',
                mask_builder:str = 'matplotlib'):
    
    """Save result into output folder.

    Args:
        image_addr (str): Address of image.
        trace_data (list): A list of lists. Each list contains trace data.
        file_name (int): File name in integer.
        output_addresses (list): A list of addresses.
        image_ext (str): Image extension.
        mask_ext (str): Mask extension.
        colorize (bool): If True, colorize the image.
        save_data_dpi (int): DPI of saved data. Default: 300.
        resize_mask_to_image_size (bool): If True, resize mask to the same size of image.
        resize_lib (str): Library used for resizing. Default: pillow. [pillow, opencv]
        mask_builder (str): Library used for mask building. Default: matplotlib. [matplotlib, pillow, opencv]
    
    Returns:
        None
    """
    img_file_name = file_name_fixer(file_name, image_ext)
    output_addr, image_output_addr, mask_output_addr, show_output_addr, temp_addr = output_addresses
    
    # Copy original image
    destination_addr = shutil.copy(image_addr, temp_addr)
    os.rename(destination_addr, Path(image_output_addr, img_file_name))


    if mask_builder == 'matplotlib':
        matplotlib_mask_builder(image_addr, trace_data, file_name, 
                                mask_output_addr, show_output_addr, 
                                mask_ext, colorize, save_data_dpi, 
                                resize_mask_to_image_size, resize_lib)

    elif mask_builder == 'pillow':
        pillow_mask_builder(image_addr, trace_data, file_name, 
                            mask_output_addr, show_output_addr, 
                            mask_ext, colorize)

    elif mask_builder == 'opencv':
        opencv_mask_builder(image_addr, trace_data, file_name, 
                            mask_output_addr, show_output_addr, 
                            mask_ext, colorize)

