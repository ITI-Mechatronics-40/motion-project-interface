
def visualize_faces(img):
	img [...,1:] = 0
	return img

def visualize_action(img):
	img[...,0] = 0
	img[...,2] = 0
	return img

def visualize_pose(img):
	img[...,:-1] = 0
	return img