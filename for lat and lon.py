def location(img_x,img_y,lat1,lon1,box_x,box_y,ratio,north_theta):

    Y = (box_y - img_y)
    X = (box_x - img_x)

    if X>0 and Y>0:
        theta = north_theta + 90 - math.degrees(math.atan(Y/X))
    elif X>0 and Y<0:
        theta = north_theta + 90 + math.degrees(math.atan(-(Y/X)))
    elif X<0 and Y<0:
        theta = north_theta + 270 - math.degrees(math.atan(Y/X))
    elif X<0 and Y>0:
        theta = north_theta + 270 + math.degrees(math.atan(-(Y/X)))
    
    d = distance(box_x,box_y,img_x,img_y)*ratio  # in Km ratio 

    origin = geopy.Point(lat1, lon1)
    destination = VincentyDistance(kilometers=d).destination(origin, theta)

    lat2, lon2 = destination.latitude, destination.longitude

    return (lat2,lon2)

    new = (lat2,lon2)
    old = (lat,lon)
    distance.distance(new,old).km
