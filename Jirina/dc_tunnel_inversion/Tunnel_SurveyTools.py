
import numpy
from SimPEG.EM.Static.DC import RxDC
from SimPEG.EM.Static.DC import SrcDC
from SimPEG.EM.Static.DC import SurveyDC
from SimPEG.EM.Static import DC, Utils as DCUtils

def lengthXY (x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    r = numpy.sqrt(dx**2 + dy**2)
    return r

def lengthXYZ (x1, y1, z1, x2, y2, z2):
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    r = numpy.sqrt(dx**2 + dy**2 + dz**2)
    return r

def getEqualySpacedPoints(endPoints, spacing):
    x1, y1, z1 = endPoints[0]
    x2, y2, z2 = endPoints[1]
    len = lengthXYZ(x1, y1, z1, x2, y2, z2)
    dirx = (x2 - x1) / len
    diry = (y1 - y2) / len
    dirz = (z2 - z1) / len
    pnum = int(numpy.floor(len / spacing)) + 1
    px = numpy.linspace(x1, x2, pnum)
    py = numpy.linspace(y1, y2, pnum)
    pz = numpy.linspace(z1, z2, pnum)
    points = numpy.c_[px, py, pz]
    return points



def _getSimpleLinearArraySurvey(points):
    sources = []
    for ii in range(points.shape[0] - 3):
        Apos = points[ii]
        Bpos = points[ii + 1]
        Mpos = points[ii + 2:-1]
        Npos = points[ii + 3:]
        receiver = DC.Rx.Dipole(Mpos, Npos)
        source = DC.Src.Dipole([receiver], Apos, Bpos)
        sources.append(source)
    survey = DC.Survey(sources)
    return survey
    pass

def _getDipoleDipoleConfigurationSurvey(points):
    sources = []
    pb = points.shape[0]
    for ii in range(pb - 3):
        Apos = points[ii]
        Bpos = points[ii + 1]
        Mpos = points[ii + 2:-1]
        Npos = points[ii + 3:]
        if ii < pb - 4 :
            Mpos = numpy.vstack([Mpos, points[pb - 3]])
            Npos = numpy.vstack([Npos, points[pb - 1]])
        receiver = DC.Rx.Dipole(Mpos, Npos)
        source = DC.Src.Dipole([receiver], Apos, Bpos)
        sources.append(source)
    survey = DC.Survey(sources)
    return survey
    pass

def _getDipoleDipoleFullCombinationSurvey(points):
    pass

def getLinearArraySurvey(endPoints, spacing, type=None):
    """
    Creates linear array dipole-dipole Survey
    :param endPoints:  numpy array [x1 y1 z1 x2 y2 z2] with survey end points position
    :return:
    """
    if type == None:
        points = getEqualySpacedPoints(endPoints, spacing)
        survey = _getSimpleLinearArraySurvey(points)
    elif type == 1:
        points = getEqualySpacedPoints(endPoints, spacing)
        survey = _getDipoleDipoleConfigurationSurvey(points)
        pass
    elif type == 2:
        raise ValueError('Survey type {n} - not implemented yet!'.format(n=type))
        pass
    elif type == 3:
        raise ValueError('Survey type {n} - not implemented yet!'.format(n=type))
        pass
    elif type == 4:
        raise ValueError('Survey type {n} - not implemented yet!'.format(n=type))
        pass
    elif type == 5:
        raise ValueError('Survey type {n} - not implemented yet!'.format(n=type))
        pass
    else:
        raise ValueError('Illegal survey type {n}!'.format(n=type))
        pass
    return points, survey


def getSurveyComposite(surveys):
    sources = []
    for ii in range(len(surveys)):
        sources.append(surveys[ii].srcList)
    survey = DC.Survey(sources)
    return survey

def _printSurveyElectrodePosition(survey):
    print("number of surveys ", survey.nSrc)
    for ii in range(survey.nSrc):
        print()
        print("survey ", ii)
        src = (survey.srcList[ii])
        print("source location", src.loc[0], src.loc[1])
        print("number of receivers ", len(src.rxList))
        for jj in range(len(src.rxList)):
            rx = src.rxList[jj]
            print("    ", rx.locs.shape)
            for kk in range(rx.locs.shape[1]):
                pass
                print("    ", rx.locs[0, kk], rx.locs[1, kk])


def _runAsMain():
    print("-- Tunnel_SurveyTools test code start")
    a = numpy.array([[0, 0, 0], [50, 0, 0]])
    allPoints, survey = getLinearArraySurvey(a, 5, 1)
    #survey = DCUtils.gen_DCIPsurvey(a, "dipole-dipole", dim=3, a=5, b=5, n=8)
    print(allPoints)
    print()
    _printSurveyElectrodePosition(survey)
    print("-- Tunnel_SurveyTools test code stop")


if __name__ == "__main__":
    _runAsMain()