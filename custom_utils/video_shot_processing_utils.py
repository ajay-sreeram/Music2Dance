import tensorflow as tf
from mobilenet import mobilenet_v1
import numpy as np
import math

'''
FOR LARGE VIDEO FILE
how to use:
-------------------------------------------------
import numpy as np
from PIL import Image
from _video_shot_processing import PersonDetection

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

detector = PersonDetection()
image_path = "person2.jpg"
image = Image.open(image_path)
image_np = load_image_into_numpy_array(image)
images_np_cropped = detector.get_person(np.expand_dims(image_np,0))

resulted cropped images is images_np_cropped[0]
-------------------------------------------------
'''
class PersonDetection:
    def __init__(self):
        self.MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
        self.PATH_TO_CKPT = "Object_detection/"+self.MODEL_NAME + '/frozen_inference_graph.pb'
        self.box_increase_percentage = .08
        # updating graph
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
    
    # images is numpy array of shape (num of images, some height, width, 3)
    def _run_detection_inference_for_single_image(self, images):
        with self.detection_graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)                

                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                        feed_dict={image_tensor: images})
                for out_index in range(output_dict['detection_classes'].shape[0]):
                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    #output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_classes'][out_index] = output_dict['detection_classes'][out_index].astype(np.uint8)
                    #output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    #output_dict['detection_scores'] = output_dict['detection_scores'][0]
                
        return output_dict

    # images is numpy array of shape (num of images, some height, width, 3)
    def _get_person_detection_coords(self, images_np):
        output_dict = self._run_detection_inference_for_single_image(images_np)
        min_score_thresh = .5
        result = []
        result_isperson_detected = []
        for image_index in range(images_np.shape[0]):
            sorted_score_index = np.argsort(output_dict['detection_scores'][image_index])
            person_box = []
            for i in range(1, len(sorted_score_index)):
                index = sorted_score_index[-i]
                # for person class is 1
                if output_dict['detection_scores'][image_index][index]>min_score_thresh and output_dict['detection_classes'][image_index][index] == 1:
                    person_box = output_dict['detection_boxes'][image_index][index]
                    break
            if len(person_box) == 0:
                result.append(())
                result_isperson_detected.append(False)
                continue
            ymin, xmin, ymax, xmax = person_box
            (im_height, im_width) = (images_np[image_index].shape[0], images_np[image_index].shape[1]) 
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
            result.append( (int(math.floor(left)), int(math.floor(right)), int(math.floor(top)), int(math.floor(bottom))))
            result_isperson_detected.append(True)
        return (np.array(result), np.array(result_isperson_detected))
    
    # image is numpy array of shape (num_images, some height, width, 3)
    def get_person(self, images_np):
        (results, results_isperson_detected) = self._get_person_detection_coords(images_np)
        images_np_cropped = []
        for index in range(results.shape[0]):
            result = results[index]
            if results_isperson_detected[index] == False:
                images_np_cropped.append(images_np[index])
                continue
            (left, right, top, bottom) = result
            # increase detected box size in all directions by 8% just to make sure complete person is in box
            width = images_np[index].shape[1]
            height = images_np[index].shape[0]
            
            (left, right, top, bottom) = self.update_box_coords(left, right, top, bottom, width, height)
            images_np_cropped.append( images_np[index, top:bottom, left:right, :])

        return (images_np_cropped, results_isperson_detected)
    
    def update_box_coords(self, left, right, top, bottom, width, height):        
        left = left - int(width*self.box_increase_percentage)
        right = right + int(width*self.box_increase_percentage)
        top = top - int(height*self.box_increase_percentage)
        bottom = bottom + int(height*self.box_increase_percentage)
        if left<0:
            left = 0
        if top<0:
            top = 0
        if right>width:
            right = width
        if bottom>height:
            bottom = height
        return (left, right, top, bottom)


'''
How to use:
--------------------------------------------------
from _video_shot_processing import PoseDetection
pose = PoseDetection()
image_pose_points = pose.get_pose(image_np)


plt.imshow(image_np)

for point in np.ceil(image_pose_points):
    plt.scatter(x=[point[1]], y=[point[0]], c='r', s=10)

partEdges = [
    [5, 6], [5,7], [7,9], [6,8], [8,10], [5,11], [6,12], 
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
]

for edge in partEdges:
    p1 = image_pose_points[edge[0]]
    p2 = image_pose_points[edge[1]]
    plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'ro-')

plt.show()

'''
class PoseDetection:
    def __init__(self):
        self.pose_graph = tf.Graph()
        #self.pose_session = tf.Session()

        with self.pose_graph.as_default():
            self.saver = tf.train.import_meta_graph('PoseNet/converter/checkpoints/model.ckpt.meta')
            #self.saver.restore(self.pose_session,tf.train.latest_checkpoint('PoseNet/converter/checkpoints/'))
        
        self.pose_session = tf.Session(graph=self.pose_graph)
        self.saver.restore(self.pose_session,tf.train.latest_checkpoint('PoseNet/converter/checkpoints/'))            

    
    def _argmax2d(self, inputs):
        height, width, depth = inputs.shape
        reshaped = inputs.reshape([height * width, depth])
        coords = reshaped.argmax(0)
        yCoords = np.expand_dims(np.floor_divide(coords, width), 1)    
        xCoords = np.expand_dims(np.mod(coords, width), 1)
        return np.concatenate((yCoords, xCoords), 1)
        
    def _getOffsetVectors(self, heatMapCoordsBuffer,offsetsBuffer):
        result = []

        for keypoint in range(17):
            heatmapY = heatMapCoordsBuffer[keypoint, 0]
            heatmapX = heatMapCoordsBuffer[keypoint, 1]
            
            y = offsetsBuffer[heatmapY, heatmapX, keypoint]
            x = offsetsBuffer[heatmapY, heatmapX, keypoint+17]
            
            result.append(np.array([y, x]))

        return np.array(result)
    
    def _getOffsetPoints(self, heatMapCoords, outputStride, offsets):
        offsetVectors = self._getOffsetVectors(heatMapCoords, offsets)
        return heatMapCoords * outputStride + offsetVectors
    
    def _getPointsConfidence(self, heatmapScores, heatMapCoords):
        result = []
        for keypoint in range(17):
            y = heatMapCoords[keypoint, 0]
            x = heatMapCoords[keypoint, 1]
            result.append(heatmapScores[y, x, keypoint])
        return np.array(result)
        
    # array of input_image numpy array
    def get_pose(self, images_np):
        images_np = images_np * (2.0 / 255.0) - 1.0
        
        sess = self.pose_session
        with self.pose_graph.as_default() as graph:
            #with self.pose_session as sess:
            #with tf.Session() as sess:
                #self.saver.restore(sess,tf.train.latest_checkpoint('PoseNet/converter/checkpoints/'))
            image = graph.get_tensor_by_name("image:0")
            offsets = graph.get_tensor_by_name("offset_2:0")
            displacementFwd = graph.get_tensor_by_name("displacement_fwd_2:0")
            displacementBwd = graph.get_tensor_by_name("displacement_bwd_2:0")
            heatmaps = graph.get_tensor_by_name("heatmap:0")
            heatmaps_result,offsets_result,_,_ = sess.run(
                [heatmaps,offsets,displacementFwd,displacementBwd], 
                feed_dict={ image: images_np } )
        
        results = []
        scores = []
        for image_index in range(images_np.shape[0]):
            heatmaps_result = heatmaps_result[image_index]
            offsets_result = offsets_result[image_index]
            heatmapValues = self._argmax2d(heatmaps_result)
            # outputStride used 16 in tensorflow.js human pose detection posenet 
            outputStride = 16 
            # offsetPoints are 'y':offsetPoints[index, 0], 'x':offsetPoints[index, 1] axis
            offsetPoints = self._getOffsetPoints(heatmapValues, outputStride,  offsets_result)
            results.append(offsetPoints.astype(int))
            # for now ignore scores(keypointConfidence)
            keypointConfidence = self._getPointsConfidence(heatmaps_result, heatmapValues)
            avg_pose_score = sum(keypointConfidence)/len(keypointConfidence)
            scores.append(avg_pose_score)
        
        return (np.array(results), np.array(scores))
