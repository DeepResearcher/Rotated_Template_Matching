
import cv2
import numpy as np

# Path
star_map = ("Star Map Path.....")
template_path = ("template path....")

img = cv2.imread(star_map)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template_org = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
w, h = template_org.shape[::-1]


## Treshold value = 0.6
def MatchTemplate(im1,im2):
    result = cv2.matchTemplate(im1, im2, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= 0.6)
    y,x = np.unravel_index(result.argmax(), result.shape)
    return result,loc,y,x

# Image rotation
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

# plot founded matches on star map
def plot_matches(img,loc):
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

    cv2.imshow("img", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Find the template on star map
def find_template(gray_img,template_org):

    result_indices = []
    rotate = 360
    #indice = 0
    for i in range(rotate):
        template = rotate_image(template_org,i)
        result,loc,y,x = MatchTemplate(gray_img, template)
        indice = result[y,x]
        result_indices.append((indice,i))
        print(indice)
        print(i)
        template = template_org
        

    indices = [i for i, x in enumerate(result_indices) if x == max(result_indices)]

    template = template_org
    angle = indices[0]
    template = rotate_image(template,angle)
    result,loc,y,x = MatchTemplate(gray_img, template)
    
    print("Best matches is",max(result_indices,key=lambda item:item[0]), "Location on Map=", (y,x))

    return result,loc



# This script return location the best template matches
result,loc = find_template(gray_img,template_org)

plot_matches(img,loc)






