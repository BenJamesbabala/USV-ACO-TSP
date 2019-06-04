import collections
from shapely.geometry import Polygon,Point  # 比较多边形交叉


Point = collections.namedtuple("Point", ["x", "y"])
OffsetCoord = collections.namedtuple("OffsetCoord", ["row", "col"])
SquareProperty = collections.namedtuple("SquareProperty",
                                        ["offsetCoord", "centerPoint", "cornerPoints", "weight", "squareSize",
                                         "isNavigonal"])
square_directions = [OffsetCoord(1, 1), OffsetCoord(1, -1), OffsetCoord(-1, -1), OffsetCoord(-1, 1)]
square_Neighbordirections = [OffsetCoord(0, -1), OffsetCoord(0, 1), OffsetCoord(-1, 0), OffsetCoord(1, 0),
                             OffsetCoord(1, 1), OffsetCoord(1, -1), OffsetCoord(-1, -1), OffsetCoord(-1, 1)]


# 得到某区域正方形网格的所有属性
def SquarePro(ullong, ullati, squaresize, squareColumn, squareRow):
    savesquareprolist = []
    rowCount = 0
    while rowCount < squareRow:
        centerY = ullati - (rowCount + 0.5) * squaresize  # 中心点纬度
        columnCount = 0
        while columnCount < squareColumn:
            centerX = ullong + (columnCount + 0.5) * squaresize  # 中心点经度
            cornerlist = GetSquareCorners(centerX, centerY, squaresize)
            squarepro = SquareProperty(OffsetCoord(rowCount, columnCount), Point(centerX, centerY), cornerlist, 1,
                                       squaresize, True)
            savesquareprolist.append(squarepro)
            columnCount = columnCount + 1
        rowCount = rowCount + 1
    return savesquareprolist


def intersection_recognition(singleCornerslist, lons, lats):
    p1 = Polygon([(singleCornerslist[0].x, singleCornerslist[0].y), (singleCornerslist[1].x, singleCornerslist[1].y),
                  (singleCornerslist[2].x, singleCornerslist[2].y), (singleCornerslist[3].x, singleCornerslist[3].y)])
    i = 0
    polygonlist = []
    while i < len(lats):
        t = (lons[i], lats[i])
        polygonlist.append(t)
        i = i + 1
    p2 = Polygon(polygonlist)
    return p1.intersects(p2)

# 得到某方向的一个点
def GetCorner(centerX, centerY, squaresize, cornerCount):
    cornerX = centerX + squaresize / 2 * square_directions[cornerCount].row
    cornerY = centerY + squaresize / 2 * square_directions[cornerCount].col
    return Point(cornerX, cornerY)

def getCenterPoint(ullong, ullati, squaresize, offsetcoord):
    rowCount = offsetcoord.row
    columnCount = offsetcoord.col
    centerY = ullati - (rowCount + 0.5) * squaresize  # 中心点纬度
    centerX = ullong + (columnCount + 0.5) * squaresize  # 中心点经度
    return Point(centerX, centerY)

# 得到一个正方形四个点的坐标
def GetSquareCorners(centerX, centerY, squaresize):
    cornerSum = 4
    cornerCount = 0
    cornerPoints = []
    while cornerCount < cornerSum:
        Points = GetCorner(centerX, centerY, squaresize, cornerCount)
        cornerPoints.append(Points)
        cornerCount = cornerCount + 1
    return cornerPoints




def getweight(squareCoord, DisNavigonal):
    results = [square_neighbor(squareCoord, 0), square_neighbor(squareCoord, 1), square_neighbor(squareCoord, 2), \
               square_neighbor(squareCoord, 3), square_neighbor(squareCoord, 4), square_neighbor(squareCoord, 5), \
               square_neighbor(squareCoord, 6), square_neighbor(squareCoord, 7)]
    neighbourDisnaCount = 0;
    for x in results:
        if (x in DisNavigonal):
            neighbourDisnaCount += 1
    if (neighbourDisnaCount > 0):
        return 0.5 * (2 ** (neighbourDisnaCount - 1))
    else:
        return 0

def square_neighbor(squareoffset, direction):
    # print(hex_add(hexcube, hex_direction(direction)))
    return square_add(squareoffset, square_Neighbordirections[direction])


def square_add(a, b):
    return OffsetCoord(a.row + b.row, a.col + b.col)
