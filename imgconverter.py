from pyciede2000 import ciede2000
from PIL import Image
import math
import numpy as np

paint = [[222, 165, 164], [214, 145, 136], [173, 111, 105], [128, 64, 64], [77, 0, 0], [77, 25, 0], [128, 0, 0], [144, 30, 30], [186, 1, 1], [179, 54, 54], [179, 95, 54], [255, 0, 0], [216, 124, 99], [255, 64, 64], [255, 128, 128], [255, 195, 192], [195, 153, 83], [128, 85, 64], [128, 106, 64], [77, 51, 38], [77, 51, 0], [128, 42, 0], [155, 71, 3], [153, 101, 21], [213, 70, 0], [218, 99, 4], [255, 85, 0], [237, 145, 33], [255, 179, 31], [255, 128, 64], [255, 170, 128], [255, 212, 128], [181, 179, 92], [77, 64, 38], [77, 77, 0], [128, 85, 0], [179, 128, 7], [183, 162, 20], [179, 137, 54], [238, 230, 0], [255, 170, 0], [255, 204, 0], [255, 255, 0], [255, 191, 64], [255, 255, 64], [223, 190, 111], [255, 255, 128], [234, 218, 184], [199, 205, 144], [128, 128, 64], [77, 77, 38], [64, 77, 38], [128, 128, 0], [101, 114, 32], [141, 182, 0], [165, 203, 12], [179, 179, 54], [191, 201, 33], [206, 255, 0], [170, 255, 0], [191, 255, 64], [213, 255, 128], [248, 249, 156], [253, 254, 184], [135, 169, 107], [106, 128, 64], [85, 128, 64], [51, 77, 38], [51, 77, 0], [67, 106, 13], [85, 128, 0], [42, 128, 0], [103, 167, 18], [132, 222, 2], [137, 179, 54], [95, 179, 54], [85, 255, 0], [128, 255, 64], [170, 255, 128], [210, 248, 176], [143, 188, 143], [103, 146, 103], [64, 128, 64], [38, 77, 38], [25, 77, 0], [0, 77, 0], [0, 128, 0], [34, 139, 34], [3, 192, 60], [70, 203, 24], [54, 179, 54], [54, 179, 95], [0, 255, 0], [64, 255, 64], [119, 221, 119], [128, 255, 128], [64, 128, 85], [64, 128, 106], [38, 77, 51], [0, 77, 26], [0, 77, 51], [0, 128, 43], [23, 114, 69], [0, 171, 102], [28, 172, 120], [11, 218, 81], [0, 255, 85], [80, 200, 120], [64, 255, 128], [128, 255, 170], [128, 255, 212], [168, 227, 189], [110, 174, 161], [64, 128, 128], [38, 77, 64], [38, 77, 77], [0, 77, 77], [0, 128, 85], [0, 166, 147], [0, 204, 153], [0, 204, 204], [54, 179, 137], [54, 179, 179], [0, 255, 170], [0, 255, 255], [64, 255, 191], [64, 255, 255], [128, 255, 255], [133, 196, 204], [93, 138, 168], [64, 106, 128], [38, 64, 77], [0, 51, 77], [0, 128, 128], [0, 85, 128], [0, 114, 187], [8, 146, 208], [54, 137, 179], [33, 171, 205], [0, 170, 255], [100, 204, 219], [64, 191, 255], [128, 212, 255], [175, 238, 238], [64, 85, 128], [38, 51, 77], [0, 26, 77], [0, 43, 128], [0, 47, 167], [54, 95, 179], [40, 106, 205], [0, 127, 255], [0, 85, 255], [49, 140, 231], [73, 151, 208], [64, 128, 255], [113, 166, 210], [100, 149, 237], [128, 170, 255], [182, 209, 234], [146, 161, 207], [64, 64, 128], [38, 38, 77], [0, 0, 77], [25, 0, 77], [0, 0, 128], [42, 0, 128], [0, 0, 205], [54, 54, 179], [95, 54, 179], [0, 0, 255], [28, 28, 240], [106, 90, 205], [64, 64, 255], [133, 129, 217], [128, 128, 255], [177, 156, 217], [150, 123, 182], [120, 81, 169], [85, 64, 128], [106, 64, 128], [51, 38, 77], [51, 0, 77], [85, 0, 128], [137, 54, 179], [85, 0, 255], [138, 43, 226], [167, 107, 207], [127, 64, 255], [191, 64, 255], [148, 87, 235], [170, 128, 255], [153, 85, 187], [140, 100, 149], [128, 64, 128], [64, 38, 77], [77, 38, 77], [77, 0, 77], [128, 0, 128], [159, 0, 197], [179, 54, 179], [184, 12, 227], [170, 0, 255], [255, 0, 255], [255, 64, 255], [213, 128, 255], [255, 128, 255], [241, 167, 254], [128, 64, 106], [105, 45, 84], [77, 38, 64], [77, 0, 51], [128, 0, 85], [162, 0, 109], [179, 54, 137], [202, 31, 123], [255, 0, 170], [255, 29, 206], [233, 54, 167], [207, 107, 169], [255, 64, 191], [218, 112, 214], [255, 128, 213], [230, 168, 215], [145, 95, 109], [128, 64, 85], [77, 38, 51], [77, 0, 25], [128, 0, 42], [215, 0, 64], [179, 54, 95], [255, 0, 127], [255, 0, 85], [255, 0, 40], [222, 49, 99], [208, 65, 126], [215, 59, 62], [255, 64, 127], [249, 90, 97], [255, 128, 170], [17, 17, 17], [34, 34, 34], [51, 51, 51], [68, 68, 68], [85, 85, 85], [102, 102, 102], [119, 119, 119], [136, 136, 136], [153, 153, 153], [170, 170, 170], [187, 187, 187], [204, 204, 204], [221, 221, 221], [238, 238, 238], [255, 255, 255]]

paint = [tuple(i)for i in paint]
lab = []
laber=0
mlen = 0






def main():
	
	global paint
	global lab
	global mlen
	img = "image.jpg"
	out = "output.jpg"
	f = 10
	global laber
	laber  = rgb_to_lab
	color_states = 4
	lab = [laber(i) for i in paint.copy()]
	
	
	"""lab = [laber(round(i*255/(color_states-1))) for i in range(color_states)]
	temp = [round(i*255/(color_states-1)) for i in range(color_states)]
	paint = []
	for r in temp.copy():
		for g in temp.copy():
			for b in temp.copy():
				paint.append((r,g,b))
	lab = [laber(i) for i in paint.copy()]"""
	mlen=len(paint)
	
	
	w,h,pixel = openImg(img)

	"""
	a = [round(i*255/(color_states-1)) for i in range(color_states)]
	def s(x):
		t = [abs(i-x) for i in a.copy()]
		s = a[t.index(min(t))]
		#print(s,x)
		return s
	largeImg = []
	img = [[tuple(map(s,pixel[i+j])) for i in range(w)]for j in range(0,h*w,w)]
	for y in range(h):
		new=[]
		for x in range(w):
			new.extend([img[y][x]]*f)
		largeImg.extend(new*f)
	
	print("done procesing",len(largeImg)**0.5, [])
	#check(img)
	makeImg(w*f,h*f,largeImg,out)
	print("done creating img :D")#


	
	
	
	return """
	print(f"done opening h:{h} w:{w}\nprocesing pls wait", len(pixel)**0.5)
	largeIm=[]
	for y in range(0,h*w,w):
		new=[]
		for x in range(w):
			new.extend([pixel[y+x]]*f)
		largeIm.extend(new*f)
	makeImg(w*f,h*f, largeIm,"original.jpg")
	
	#img=[getClosestCol(pixel[y+x]) for y in range(h) for _ in range(f) for x in range(w) for _ in range(f)]
	img = [[tuple(paint[closestColCiede20002(pixel[i+j])]) for i in range(w)]for j in range(0,h*w,w)]
	#img = [[tuple(paint[highbreed(pixel[i+j])]) for i in range(w)]for j in range(0,h*w,w)]
	
	
	
	count=0
	print("done replacing into new color :D",len(img))
	makeImg(w,h,[img[j][i] for j in range(h) for i in range(w)],"1"+out)
	largeImg=[]
	for y in range(h):
		new=[]
		for x in range(w):
			new.extend([img[y][x]]*f)
		largeImg.extend(new*f)
	
	print("done procesing",len(largeImg)**0.5, [])
	#check(img)
	makeImg(w*f,h*f,largeImg,out)
	print("done creating img :D")

def highbreed (col):
    #this is just experiment lmao
    #this is as it matches the color but it's very noticeable if you zoom it pixelize
    L1,a1,b1=rgb_to_lab(col)
    score = 99999999
    id= 0
    for i in range(255):
        deltaE = math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(col, paint[i])))
        
        L2, a2, b2 = lab[i]

        delta_L = L2 - L1
        delta_a = a2 - a1
        delta_b = b2 - b1

        C1 = math.sqrt(a1**2 + b1**2)
        C2 = math.sqrt(a2**2 + b2**2)
        delta_C = C2 - C1

        delta_H = (delta_a**2 + delta_b**2 - delta_C**2)*0.5
        

        SC = 1 + 0.045 * C1
        SH = 1 + 0.015 * C1

        delta_E = (delta_L**2 + (delta_C / (SC*0.939))**2 + (delta_H /(SH*3))**2)*0.5
        delta_E+=13
        deltaE+= 61
        #delta_E = 0 if delta_E < 0 else math.sqrt(delta_E)
        #will check if cie76 is relatively close to cir94
        if ((delta_E**2)+(deltaE**2))**0.5*(+delta_E)/(1+deltaE) <= score:
        	score = delta_E
        	id = i
    return id


def closestColCie76(color1):
    # CIE76 color distance
    #print(color1,color2)
    score = 9999999
    id = 0
    #print(color1)
    l,a,b = laber(color1)
    for i in range(mlen):
        L,A,B=lab[i]
        deltaE = ((l-L)**2+(a-A)**2+(b-B)**2)
        if deltaE <= score:
        	score = deltaE
        	id = i
    return id

def closestColCie94(color1):
    L1, a1, b1 = laber(color1)
    score = 9999999999
    id = 0
    for i in range(mlen):
        L2, a2, b2 = lab[i]

        delta_L = L2 - L1
        delta_a = a2 - a1
        delta_b = b2 - b1

        C1 = math.sqrt(a1**2 + b1**2)
        C2 = math.sqrt(a2**2 + b2**2)
        delta_C = C2 - C1

        delta_H = (delta_a**2 + delta_b**2 - delta_C**2)*0.5
        

        SC = 1 + 0.045 * C1
        SH = 1 + 0.015 * C1

        delta_E = (delta_L**2 + (delta_C / SC)**2 + (delta_H / SH)**2)*0.5
        if delta_E <= score:
        	score = delta_E
        	id = i
    return id
    


def closestColCiede2000(col):
	d = [ciede2000(val,laber(col))["delta_E_00"] for val in lab.copy()]
	return d.index(min(d))
 
def closestColCiede20002(col):

	d = [delta_E_00(laber(col),i) for i in lab.copy()]
	return d.index(min(d))
def delta_E_00(color1, color2):
    # Calculate LAB values for the two colors
    lab_color2 = color1
    lab_color1 = color2

    # Calculate CIEDE2000 color difference
    delta_l = lab_color2[0] - lab_color1[0]
    delta_a = lab_color2[1] - lab_color1[1]
    delta_b = lab_color2[2] - lab_color1[2]

    c1 = math.sqrt(lab_color1[1] ** 2 + lab_color1[2] ** 2)
    c2 = math.sqrt(lab_color2[1] ** 2 + lab_color2[2] ** 2)
    delta_c = c2 - c1
    
    delta_h = (delta_a ** 2 + delta_b ** 2 - delta_c ** 2)
    delta_h = 0 if delta_h <0 else delta_h**0.5

    sl = 1
    sc = 1.0 + 0.045 * c1
    sh = 1.0 + 0.015 * c1

    delta_theta = math.atan2(lab_color2[2], lab_color2[1]) - math.atan2(lab_color1[2], lab_color1[1])

    if delta_theta > math.pi:
        delta_theta -= 2.0 * math.pi
    elif delta_theta < -math.pi:
        delta_theta += 2.0 * math.pi

    delta_theta = math.degrees(delta_theta)

    if delta_theta < 0.0:
        delta_theta += 360.0

    delta_theta = 1.0 + 0.015 * c1 * math.exp(-((delta_theta - 275.0) / 25.0) ** 2)

    rt = -2.0 * ((c1 ** 7 / (c1 ** 7 + 25.0 ** 7))**0.5)

    a= (
        (delta_l / (sl)) ** 2 +
        (delta_c / (sc)) ** 2 +
        (delta_h / (sh)) ** 2 +
        rt * (delta_c / (sc)) * (delta_h / (sh))
    )**0.5
    return a if type(a)!=type(complex(1)) else 99999



def _convert_gamma(value):
	return value / 12.92 if value <= 0.04045 else (value + 0.055) / 1.055**2.4

def _convert_xyz(value):
	return value**1/3 if value > 0.008856 else (value * 903.3 + 16) / 116
def f(t):
    """
    if t > 0.008856:
        return t ** (1/3)
    else:
        return 7.787 * t + 16/116"""
    return t**(1/3) if t > 0.008856 else 7.787*t+16/116
            
def openImg(imgPath):
	img = Image.open(imgPath)
	w,h = img.size
	return w,h,list(img.getdata())

def makeImg(w,h,pixel,outpath):
	img = Image.new("RGB",(w,h))
	img.putdata(pixel)
	img.save(outpath)



def closestColCiede20003(col):
	d = [CIEDE2000(laber(col),lab[i]) for i in range(mlen)]
	try:
		
		return d.index(min(d))
	except:
		print(list(d), col,mel)
		raise Error ("idk what happen here man")

def CIEDE2000(Lab_1, Lab_2):
    '''Calculates CIEDE2000 color distance between two CIE L*a*b* colors'''
    C_25_7 = 6103515625 # 25**7

    L1, a1, b1 = Lab_1
    L2, a2, b2 = Lab_2
    
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_ave = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(C_ave**7 / (C_ave**7 + C_25_7)))

    L1_, L2_ = L1, L2
    a1_, a2_ = (1 + G) * a1, (1 + G) * a2
    b1_, b2_ = b1, b2

    C1_ = math.sqrt(a1_**2 + b1_**2)
    C2_ = math.sqrt(a2_**2 + b2_**2)

    if b1_ == 0 and a1_ == 0:
    	h1_ = 0
    elif a1_ >= 0:
    	h1_ = math.atan2(b1_, a1_)
    else:
    	h1_ = math.atan2(b1_, a1_) + 2 * math.pi

    if b2_ == 0 and a2_ == 0:
    	h2_ = 0
    elif a2_ >= 0:
    	h2_ = math.atan2(b2_, a2_)
    else:
    	h2_ = math.atan2(b2_, a2_) + 2 * math.pi

    dL_ = L2_ - L1_
    dC_ = C2_ - C1_    
    dh_ = h2_ - h1_
    if C1_ * C2_ == 0:
    	dh_ = 0
    elif dh_ > math.pi:
    	dh_ -= 2 * math.pi
    elif dh_ < -math.pi:
    	dh_ += 2 * math.pi        
    dH_ = 2 * math.sqrt(C1_ * C2_) * math.sin(dh_ / 2)

    L_ave = (L1_ + L2_) / 2
    C_ave = (C1_ + C2_) / 2

    _dh = abs(h1_ - h2_)
    _sh = h1_ + h2_
    C1C2 = C1_ * C2_

    if _dh <= math.pi and C1C2 != 0:
    	h_ave = (h1_ + h2_) / 2
    elif _dh  > math.pi and _sh < 2 * math.pi and C1C2 != 0:
    	h_ave = (h1_ + h2_) / 2 + math.pi
    elif _dh  > math.pi and _sh >= 2 * math.pi and C1C2 != 0:
    	h_ave = (h1_ + h2_) / 2 - math.pi 
    else:
    	h_ave = h1_ + h2_

    T = 1 - 0.17 * math.cos(h_ave - math.pi / 6) + 0.24 * math.cos(2 * h_ave) + 0.32 * math.cos(3 * h_ave + math.pi / 30) - 0.2 * math.cos(4 * h_ave - 63 * math.pi / 180)

    h_ave_deg = h_ave * 180 / math.pi
    if h_ave_deg < 0:
    	h_ave_deg += 360
    elif h_ave_deg > 360:
    	h_ave_deg -= 360
    
    dTheta = 30 * math.exp(-(((h_ave_deg - 275) / 25)**2))

    R_C = 2 * math.sqrt(C_ave**7 / (C_ave**7 + C_25_7))  
    S_C = 1 + 0.045 * C_ave
    S_H = 1 + 0.015 * C_ave * T

    Lm50s = (L_ave - 50)**2
    S_L = 1 + 0.015 * Lm50s / math.sqrt(20 + Lm50s)
    R_T = -math.sin(dTheta * math.pi / 90) * R_C

    k_L, k_C, k_H = 1, 1, 1

    f_L = dL_ / k_L / S_L
    f_C = dC_ / k_C / S_C
    f_H = dH_ / k_H / S_H
    
    return math.sqrt(f_L**2 + f_C**2 + f_H**2 + R_T * f_C * f_H)


"""__copyright__ = "Copyright 2020, Rubens Technologies"

__license__ = "MIT"
__version__ = "0.0.1" """

def rgb_to_lab(rgb):
    #D65
    r,g,b = rgb
    r,g,b = [((j + 0.055)/1.055)**2.4 if j > 0.04045 else j/12.92 for j in (r/255,g/255,b/255)]

    x = lf((r * 0.4124564 + g * 0.3575761 + b * 0.1804375)/0.950489)
    y = lf((r * 0.2126729 + g * 0.7151522 + b * 0.0721750))
    z = lf((r * 0.0193339 + g * 0.1191920 + b * 0.9503041)/1.085188)

    return [(116 * y - 16), (x - y) * 500,  (y - z) * 200]
def rgbToLab(rgb):
	r,g,b = rgb
	r,g,b = [((j + 0.055)/1.055)**2.4 if j > 0.04045 else j/12.92 for j in (r/255,g/255,b/255)]
	#D50
	X = ((r*0.4360747)+(g*0.3850649)+(b*0.1430804))
	Y = ((r*0.2225045)+(g*0.7168786)+(b*0.0606169))
	Z = ((r*0.0139322)+(g*0.0971045)+(b*0.7141733))
	Xn = 0.964242
	Zn = 0.825188
	
	return (116*lf(Y) - 16, 500*(lf(X/Xn)-lf(Y)), 200*(lf(Y)-lf(Z/Zn)))


def rgb2lab(col,C=255, illuminant='D50'):
    # Normalise values
    r = col[0]/C
    g = col[1]/C
    b = col[2]/C
    # Linearise RGB values
    r,g,b = [((j + 0.055)/1.055)**2.4 if j > 0.04045 else j/12.92 for j in (r/255,g/255,b/255)]
    rgb=(r,g,b)
    # Convert to XYZ. Need to decide what RGB we are using
    # http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    # https://en.wikipedia.org/wiki/CIELAB_color_space
    if illuminant == 'D50':
        # sRGB
        M = ([[0.4360747,  0.3850649,  0.1430804], \
                     [0.2225045,  0.7168786,  0.0606169], \
                     [0.0139322,  0.0971045,  0.7141733]])
        Xn = 0.964242
        Zn = 0.825188
    elif illuminant == 'D65':
        M = ([[0.4124564,  0.3575761,  0.1804375], \
                      [0.2126729,  0.7151522,  0.0721750], \
                      [0.0193339,  0.1191920,  0.9503041]])
        # D65 illuminant
        Xn = 0.950489
        Zn = 1.085188
    else:
        print("Error. 'D50' or 'D65' are the only allowed values for illuminant.")
        return
    XYZ = np.dot(M,rgb)
    X = XYZ[0]
    Y = XYZ[1]
    Z = XYZ[2]
    # Finally convert to Lab
    return (116*lf(Y) - 16,
    500*(lf(X/Xn)-lf(Y)),
    200*(lf(Y)-lf(Z/Zn)))

def lf(t):
    return t**(1/3) if (t>6./29**3) else t/(3*6./29**2) + 4/29
def gammaToLinear(c):
  return ((c + 0.055) / 1.055)**2.4 if c >= 0.04045 else c / 12.92
  
 
def rgbToOklab(rgb):
  r,g,b = rgb
  #This is my undersanding: JavaScript canvas and many other virtual and literal devices use gamma-corrected (non-linear lightness) RGB, or sRGB. To convert sRGB values for manipulation in the Oklab color space, you must first convert them to linear RGB. Where Oklab interfaces with RGB it expects and returns linear RGB values. This next step converts (via a function) sRGB to linear RGB for Oklab to use:
  r = gammaToLinear(r / 255)
  g = gammaToLinear(g / 255)
  b = gammaToLinear(b / 255)
  #This is the Oklab math:
  l = math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
  m = math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
  s = math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
  #Math.crb (cube root) here is the equivalent of the C++ cbrtf function here: https://bottosson.github.io/posts/oklab/#converting-from-linear-srgb-to-oklab
  return (
    l * +0.2104542553 + m * +0.7936177850 + s * -0.0040720468,
    l * +1.9779984951 + m * -2.4285922050 + s * +0.4505937099,
    l * +0.0259040371 + m * +0.7827717662 + s * -0.8086757660)
  
  
main()