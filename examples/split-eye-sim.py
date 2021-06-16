from utils import load_route_naw, image_split, display_split, rotate, cor_coef
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk", font_scale=1)
sns.set_style('white')



path = 'test_route/route1/'
route = load_route_naw(path, route_id=1, imgs=True)

memory_index = 350
img = route['imgs'][memory_index]
overlap = 30
blind = 60
left, right = image_split(img, overlap, blind)
display_split(left, right, file_name='memory')
x_deg = range(-180, 180)

left_sim = []
right_sim = []
for d in x_deg:
    rot_image = rotate(d, img)
    le, re = image_split(rot_image, overlap, blind)
    left_sim.append(cor_coef(left, le))
    right_sim.append(cor_coef(right, re))

fig = plt.figure(figsize=(10, 7))
plt.plot(x_deg, left_sim, label='Left')
plt.plot(x_deg, right_sim, label='right')
plt.ylabel('Similarity')
plt.xlabel('Degrees')
plt.legend()
plt.savefig('sims.png')
plt.show()