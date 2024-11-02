import re
import cv2
import easyocr
import numpy as np
import jellyfish
import torch

class DoorbellTask:
    def __init__(self, draw_debug = False) -> None:
        reader = easyocr.Reader(['en'])
        self.reader = reader
        self.draw_debug = draw_debug
        self.max_x = float('-inf')
        self.min_x = float('inf')
        self.max_y = float('-inf')
        self.min_y = float('inf')
        self.target_name = ""
        self.is_cuda = torch.cuda.is_available()

        self.names = ["Smith", "Jones", "Williams", "Brown", "Taylor", 
                        "Davies", "Wilson", "Evans", "Thomas", "Robert", "Johnson"]
    
    def Set_Target_Image(self,target_image):
        target_result = self._get_ocr_result(target_image)
        if len(target_result) == 1:
            found_name = target_result[0][1]
            found_name = self.__clean_and_split_text(found_name)
            if len(found_name) == 1:
                found_name = found_name[0]
                found_name = self.__find_most_similar(found_name)
                if found_name in self.names:
                    self.target_name = found_name
                    return self.target_name
                else:
                    print("Wrong Name")
                    return ""
                
        return ""
        
    
    def __preProcess_img(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, threshold1=30, threshold2=150)

        kernel = np.ones((30, 30), np.uint8)

        dilated = cv2.dilate(edges, kernel, iterations=1)

        contours_merged, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        biggest_contour = max(contours_merged, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest_contour)
        cropped_image = image[y:y+h, x:x+w]

        height, width = cropped_image.shape[:2]
        # if self.is_cuda:
        #     cropped_image = cv2.resize(cropped_image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)

        return cropped_image
    
    def __clean_and_split_text(self, text):
        words = re.split(r'[^a-zA-Z]+', text)

        words = [word for word in words if word]
        
        return words
    
    def __similar(self, a, b):
        # return Levenshtein.distance(a, b)
        # return SequenceMatcher(None, a, b).ratio()
        return jellyfish.jaro_similarity(a, b)

    def __find_most_similar(self, text):
        max_sim = float('-inf')  # Start with a large number
        most_similar_name = None
        
        for name in self.names:
            sim = self.__similar(name.lower(),text.lower())
            
            if sim > max_sim:
                max_sim = sim
                most_similar_name = name
        
        return most_similar_name

    def __calculate_iou(self, box1, box2):
        """Calculate the Intersection over Union (IoU) of two bounding boxes."""
        x1 = max(box1[0][0], box2[0][0])
        y1 = max(box1[0][1], box2[0][1])
        x2 = min(box1[2][0], box2[2][0])
        y2 = min(box1[2][1], box2[2][1])

        # Calculate the area of intersection
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate the area of both bounding boxes
        box1_area = (box1[2][0] - box1[0][0]) * (box1[2][1] - box1[0][1])
        box2_area = (box2[2][0] - box2[0][0]) * (box2[2][1] - box2[0][1])

        # Calculate IoU
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    def __merge_boxes_and_texts(self, results, threshold=0.0):
        merged_results = []
        
        for i, (bbox1, text1, prob1) in enumerate(results):
            bbox1 = np.array(bbox1).astype(int)
            merged = False

            for merged_data in merged_results:
                merged_bbox, merged_text = merged_data['bbox'], merged_data['text']
                
                # Check if the boxes overlap
                if self.__calculate_iou(bbox1, merged_bbox) > threshold:
                    # Merge bounding boxes (take the outermost coordinates)
                    merged_bbox[0][0] = min(merged_bbox[0][0], bbox1[0][0])
                    merged_bbox[0][1] = min(merged_bbox[0][1], bbox1[0][1])
                    merged_bbox[2][0] = max(merged_bbox[2][0], bbox1[2][0])
                    merged_bbox[2][1] = max(merged_bbox[2][1], bbox1[2][1])
                    
                    # Merge text
                    merged_text = merged_text + ',' + text1
                    merged_data['text'] = merged_text
                    
                    merged = True
                    break

            if not merged:
                merged_results.append({
                    'bbox': bbox1,
                    'text': text1
                })

        return merged_results
    

    def __extract_data(self, image, results):
        self.max_x = float('-inf')
        self.min_x = float('inf')

        self.max_y = float('-inf')
        self.min_y = float('inf')

        y_centers = []
        x_centers = []
        doorbells = []
        names = []
        duplicate = False

        for data in results:
            bbox = data['bbox']
            text = data['text']

            bbox = np.array(bbox)
            bbox = bbox.astype(int)
            # Draw the bounding box
            y_center = (bbox[0][1] + bbox[2][1]) // 2
            x_center = (bbox[0][0] + bbox[2][0]) // 2
            
            self.max_x = max(self.max_x, bbox[0][0])
            self.min_x = min(self.min_x, bbox[2][0])
            
            self.max_y = max(self.max_y, bbox[0][1])
            self.min_y = min(self.min_y, bbox[2][1])
            
            top_left = tuple(bbox[0])
            bottom_right = tuple(bbox[2])

            text_clean = self.__clean_and_split_text(text=text)
            result = []
            for name in text_clean:
                if name in names:
                    duplicate = True

                names.append(name)
                temp_result = self.__find_most_similar(name)
                result.append(temp_result)
            text = ','.join(result)

            if self.draw_debug:
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            
            data = {"text": text, "y_center": y_center, "x_center": x_center, "bounderis": bbox}
            
            y_centers.append(y_center)
            x_centers.append(x_center)
            doorbells.append(data)
            
        return x_centers, y_centers, doorbells, duplicate
    
    def __assign_row(self, y_center, row_boundaries):
        lowest_bound = 10000
        lowest_bound_i = -1
        
        for i, boundary in enumerate(row_boundaries):
            diff = abs(y_center - boundary)
            if diff < lowest_bound:
                lowest_bound_i = i
                lowest_bound = diff
            
        return lowest_bound_i

    def __create_boundaries(self, centers, img, axis='y'):
        centers = sorted(centers)
        diffs = np.diff(centers)

        threshold = np.mean(diffs)
        boundaries = [centers[0]]
        min_gap = 1000
        
        for i, diff in enumerate(diffs):
            if diff > threshold:
                center = centers[i + 1]
                min_gap = min(min_gap, diff)
                boundaries.append(center)  
                        
        diff_bounds = np.diff(boundaries)
        for i, bnd_diff in enumerate(diff_bounds):
            if bnd_diff > min_gap * 2:
                boundaries.insert(i, boundaries[i] + min_gap)

        if self.draw_debug:
            for bound in boundaries:
                if axis == 'y':
                    cv2.line(img, (0, bound), (img.shape[1], bound), (100, 0, 0), 2)
                else:
                    cv2.line(img, (bound, 0), (bound, img.shape[0]), (100, 0, 0), 2)
                    
        return boundaries
        
    def Find_target_doorbell(self,panel_img):
        # start_time = time.time()
        img = panel_img.copy()
        img = self.__preProcess_img(img)

        kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
        
        results = self._get_ocr_result(img)
        if len(results) == 0:
            print("No text found")
            return img,False,[-1,-1]
        
        merged_results = self.__merge_boxes_and_texts(results)
        x_centers, y_centers, doorbells, duplicate = self.__extract_data(img,merged_results)
        
        if duplicate:
            print("Duplicate detect")
            return img,False,[-1,-1]
        
        if len(y_centers) == 1:
            print("No text found")
            return img,False,[-1,-1]
        
        row_boundaries = self.__create_boundaries(y_centers, img, axis='y')
        column_boundaries = self.__create_boundaries(x_centers, img, axis='x')
        
        target_found = False
        target_col = -1
        target_row = -1

        if len(column_boundaries) != 2:
            print("Wrong Columns")
            return img, False,[-1,-1]
        
        if len(row_boundaries) != 4:
            print("Wrong Rows")
            return img, False,[-1,-1]
        
        for doorbell in doorbells:
            y_center = doorbell['y_center']
            x_center = doorbell['x_center']
            text = doorbell['text']
            
            if x_center < (self.max_x + self.min_x) / 2:
                column = 0 # left
            else:
                column = 1 # right
            row = self.__assign_row(y_center, row_boundaries) + 1
            
            doorbell['column'] = column
            doorbell['row'] = row
            
            if target_found == False and self.target_name.lower() in text.lower():
                target_found = True
                target_col = column
                target_row = row

        if self.draw_debug:
            img = self.Draw_Box(img , doorbells)

        return img,target_found,[target_col,target_row]
    
    def Draw_Box(self, image, doorbells):
        for doorbell in doorbells:
            bbox = doorbell['bounderis']
            text = doorbell['text']

            (top_left, top_right, bottom_right, bottom_left) = bbox

            top_left = tuple([int(val) for val in top_left])
            bottom_right = tuple([int(val) for val in bottom_right])

            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green box

            cv2.putText(image, text, (top_left[0], top_left[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Blue text
        
        return image
    
    def _get_ocr_result(self,img):
        results = self.reader.readtext(img)
        return results
        