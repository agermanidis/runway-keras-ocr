import runway
import keras_ocr
import numpy as np


@runway.setup
def setup():
    return keras_ocr.pipeline.Pipeline()


@runway.command('recognize', inputs={'image': runway.image}, outputs={'bboxes': runway.array(runway.image_bounding_box), 'labels': runway.array(runway.text)})
def recognize(model, inputs):
    width, height = inputs['image'].size
    predictions = model.recognize(np.array(inputs['image']))
    labels = []
    bboxes = []
    for label, bbox in predictions:
        labels.append(label)
        min_x, min_y = bbox.min(0)
        max_x, max_y = bbox.max(0)
        bboxes.append([
            min_x / width,
            min_y / height,
            max_x / width,
            max_y / height
        ])
    return {'labels': labels, 'bboxes': bboxes}


if __name__ == "__main__":
    runway.run()