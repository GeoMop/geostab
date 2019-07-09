// Physical entities
// 1 - volume - primary mesh - forward simulation
// 2 - volume - para mesh - region of inversion
// 99 - points - electrodes position
// 1 - surface - Neumann boundary - no flux
// 2 - surface - mixed boundary 


len1 = 50;
len2 = 5;
len99 = 0.5;

// uroven pocvy tunelu
dn = 0;
dz = 0;

Point( 1+dn) = {-85, 0, 0+dz, len1};
Point( 2+dn) = {-43, 0, 0+dz, len2};
Point( 3+dn) = {-40, 0, 0+dz, len99};
Point( 4+dn) = {1, 0, 0+dz, len99};
Point( 5+dn) = {5, 0, 0+dz, len2};

Point( 6+dn) = {-85, 3, 0+dz, len1};
Point( 7+dn) = {-40, 3, 0+dz, len2};
Point( 8+dn) = {-1, 3, 0+dz, len2};
Point( 9+dn) = {3, 3, 0+dz, len2};

Point(10+dn) = {-43, -20, 0+dz, len2};
Point(11+dn) = {-40, -20, 0+dz, len2};

Point(12+dn) = {-40, -15, 0+dz, len2};
Point(13+dn) = {1, -15, 0+dz, len2};
Point(14+dn) = {11, -15, 0+dz, len2};

Point(15+dn) = {-40, 15, 0+dz, len2};
Point(16+dn) = {-9, 15, 0+dz, len2};

Point(17+dn) = {51, -75, 0+dz, len1};
Point(18+dn) = {55, -75, 0+dz, len1};
Point(19+dn) = {-49, 75, 0+dz, len1};
Point(20+dn) = {-45, 75, 0+dz, len1};

//Point(21+dn) = {55, 75, 0+dz, len1};
//Point(22+dn) = {-100, 75, 0+dz, len1};
//Point(23+dn) = {-100, -75, 0+dz, len1};


// uroven stropu tunelu
dn = 30;
dz = 3;

Point( 1+dn) = {-85, 0, 0+dz, len1};
Point( 2+dn) = {-43, 0, 0+dz, len2};
Point( 3+dn) = {-40, 0, 0+dz, len99};
Point( 4+dn) = {1, 0, 0+dz, len99};
Point( 5+dn) = {5, 0, 0+dz, len2};

Point( 6+dn) = {-85, 3, 0+dz, len1};
Point( 7+dn) = {-40, 3, 0+dz, len2};
Point( 8+dn) = {-1, 3, 0+dz, len2};
Point( 9+dn) = {3, 3, 0+dz, len2};

Point(10+dn) = {-43, -20, 0+dz, len2};
Point(11+dn) = {-40, -20, 0+dz, len2};

Point(12+dn) = {-40, -15, 0+dz, len2};
Point(13+dn) = {1, -15, 0+dz, len2};
Point(14+dn) = {11, -15, 0+dz, len2};

Point(15+dn) = {-40, 15, 0+dz, len2};
Point(16+dn) = {-9, 15, 0+dz, len2};

Point(17+dn) = {51, -75, 0+dz, len1};
Point(18+dn) = {55, -75, 0+dz, len1};
Point(19+dn) = {-49, 75, 0+dz, len1};
Point(20+dn) = {-45, 75, 0+dz, len1};

//Point(21+dn) = {55, 75, 0+dz, len1};
//Point(22+dn) = {-100, 75, 0+dz, len1};
//Point(23+dn) = {-100, -75, 0+dz, len1};

// bottom inverzniho modelu - paramesh bottom
dn = 60;
dz = -15;

Point( 3+dn) = {-40, 0, 0+dz, len2};
Point( 4+dn) = {1, 0, 0+dz, len2};

//Point( 7+dn) = {-40, 3, 0+dz, len2};
//Point( 8+dn) = {-1, 3, 0+dz, len2};

//Point(10+dn) = {-43, -20, 0+dz, len2};
//Point(11+dn) = {-40, -20, 0+dz, len2};

Point(12+dn) = {-40, -15, 0+dz, len2};
Point(13+dn) = {1, -15, 0+dz, len2};

Point(15+dn) = {-40, 15, 0+dz, len2};
Point(16+dn) = {-9, 15, 0+dz, len2};

// top inverzniho modelu - paramesh top
dn = 80;
dz = 15;

Point( 3+dn) = {-40, 0, 0+dz, len2};
Point( 4+dn) = {1, 0, 0+dz, len2};

//Point( 7+dn) = {-40, 3, 0+dz, len2};
//Point( 8+dn) = {-1, 3, 0+dz, len2};

//Point(10+dn) = {-43, -20, 0+dz, len2};
//Point(11+dn) = {-40, -20, 0+dz, len2};

Point(12+dn) = {-40, -15, 0+dz, len2};
Point(13+dn) = {1, -15, 0+dz, len2};

Point(15+dn) = {-40, 15, 0+dz, len2};
Point(16+dn) = {-9, 15, 0+dz, len2};

// bottom modelu - primary mesh bottom
dn = 100;
dz = -60;

Point(18+dn) = {55, -75, 0+dz, len1};

Point(21+dn) = {55, 75, 0+dz, len1};
Point(22+dn) = {-100, 75, 0+dz, len1};
Point(23+dn) = {-100, -75, 0+dz, len1};

// bottom modelu - primary mesh bottom
dn = 130;
dz = 60;

Point(18+dn) = {55, -75, 0+dz, len1};

Point(21+dn) = {55, 75, 0+dz, len1};
Point(22+dn) = {-100, 75, 0+dz, len1};
Point(23+dn) = {-100, -75, 0+dz, len1};






//+
Line(1) = {1, 2};
//+
Line(2) = {2, 10};
//+
Line(3) = {10, 11};
//+
Line(4) = {11, 12};
//+
Line(5) = {12, 3};
//+
Line(6) = {3, 4};
//+
Line(7) = {6, 1};
//+
Line(8) = {6, 7};
//+
Line(9) = {7, 8};
//+
Line(10) = {12, 13};
//+
Line(11) = {4, 13};
//+
Line(12) = {4, 14};
//+
Line(13) = {8, 16};
//+
Line(14) = {7, 15};
//+
Line(15) = {15, 16};
//+
Line(16) = {9, 5};
//+
Line(17) = {14, 17};
//+
Line(18) = {17, 18};
//+
Line(19) = {18, 5};
//+
Line(20) = {9, 20};
//+
Line(21) = {20, 19};
//+
Line(22) = {19, 16};
//+
Line(23) = {36, 31};
//+
Line(24) = {31, 32};
//+
Line(25) = {32, 40};
//+
Line(26) = {40, 41};
//+
Line(27) = {41, 42};
//+
Line(28) = {42, 33};
//+
Line(29) = {33, 34};
//+
Line(30) = {34, 43};
//+
Line(31) = {43, 42};
//+
Line(32) = {34, 44};
//+
Line(33) = {36, 37};
//+
Line(34) = {37, 38};
//+
Line(35) = {37, 45};
//+
Line(36) = {45, 46};
//+
Line(37) = {46, 38};
//+
Line(38) = {44, 47};
//+
Line(39) = {47, 48};
//+
Line(40) = {48, 35};
//+
Line(41) = {35, 39};
//+
Line(42) = {39, 50};
//+
Line(43) = {50, 49};
//+
Line(44) = {49, 46};
//+
Line(45) = {31, 1};
//+
Line(46) = {36, 6};
//+
Line(47) = {32, 2};
//+
Line(48) = {40, 10};
//+
Line(49) = {41, 11};
//+
Line(50) = {42, 12};
//+
Line(51) = {33, 3};
//+
Line(52) = {37, 7};
//+
Line(53) = {45, 15};
//+
Line(54) = {46, 16};
//+
Line(55) = {38, 8};
//+
Line(56) = {34, 4};
//+
Line(57) = {43, 13};
//+
Line(58) = {47, 17};
//+
Line(59) = {48, 18};
//+
Line(60) = {49, 19};
//+
Line(61) = {50, 20};
//+
Line(62) = {8, 4};
//+
Line(63) = {38, 34};
//+
Line(64) = {37, 33};
//+
Line(65) = {32, 33};
//+
Line(66) = {2, 3};
//+
Line(67) = {95, 83};
//+
Line(68) = {83, 92};
//+
Line(69) = {92, 93};
//+
Line(70) = {93, 84};
//+
Line(71) = {84, 96};
//+
Line(72) = {96, 95};
//+
Line(73) = {95, 45};
//+
Line(74) = {92, 42};
//+
Line(75) = {93, 43};
//+
Line(76) = {84, 34};
//+
Line(77) = {96, 46};
//+
Line(78) = {75, 63};
//+
Line(79) = {63, 72};
//+
Line(80) = {72, 73};
//+
Line(81) = {73, 64};
//+
Line(82) = {64, 76};
//+
Line(83) = {76, 75};
//+
Line(84) = {15, 75};
//+
Line(85) = {3, 63};
//+
Line(86) = {12, 72};
//+
Line(87) = {13, 73};
//+
Line(88) = {4, 64};
//+
Line(89) = {16, 76};
//+
Line(90) = {153, 123};
//+
Line(91) = {123, 118};
//+
Line(92) = {153, 148};
//+
Line(93) = {148, 48};
//+
Line(94) = {18, 118};
//+
Line(95) = {148, 151};
//+
Line(96) = {151, 121};
//+
Line(97) = {121, 118};
//+
Line(98) = {152, 151};
//+
Line(99) = {121, 122};
//+
Line(100) = {122, 152};
//+
Line(101) = {152, 153};
//+
Line(102) = {122, 123};
//+
Line(103) = {3, 7};


//+
Curve Loop(1) = {72, 67, 68, 69, 70, 71};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {83, 78, 79, 80, 81, 82};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {31, 28, 29, 30};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {10, -11, -6, -5};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {9, 13, -15, -14};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {36, 37, -34, 35};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {34, 63, -29, -64};
//+
Plane Surface(7) = {7};

//+
Curve Loop(8) = {9, 62, -6, 103};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {72, 73, 36, -77};
//+
Plane Surface(9) = {9};
//+
Curve Loop(10) = {36, 54, -15, -53};
//+
Plane Surface(10) = {10};
//+
Curve Loop(11) = {69, 75, 31, -74};
//+
Plane Surface(11) = {11};
//+
Curve Loop(12) = {31, 50, 10, -57};
//+
Plane Surface(12) = {12};
//+
Curve Loop(13) = {86, 80, -87, -10};
//+
Plane Surface(13) = {13};
//+
Curve Loop(14) = {70, 76, 30, -75};
//+
Plane Surface(14) = {14};
//+
Curve Loop(15) = {30, 57, -11, -56};
//+
Plane Surface(15) = {15};
//+
Curve Loop(16) = {87, 81, -88, 11};
//+
Plane Surface(16) = {16};
//+
Curve Loop(17) = {76, -63, -37, -77, -71};
//+
Plane Surface(17) = {17};
//+
Curve Loop(18) = {37, 55, 13, -54};
//+
Plane Surface(18) = {18};
//+
Curve Loop(19) = {13, 89, -82, -88, -62};
//+
Plane Surface(19) = {19};
//+
Curve Loop(20) = {67, 68, 74, 28, -64, 35, -73};
//+
Plane Surface(20) = {20};
//+
Curve Loop(21) = {35, 53, -14, -52};
//+
Plane Surface(21) = {21};
//+
Curve Loop(22) = {5, -51, -28, 50};
//+
Plane Surface(22) = {22};
//+
Curve Loop(23) = {5, 85, 79, -86};
//+
Plane Surface(23) = {23};
//+
Curve Loop(24) = {103, 14, 84, 78, -85};
//+
Plane Surface(24) = {24};
//+
Curve Loop(25) = {26, 49, -3, -48};
//+
Plane Surface(25) = {25};
//+
Curve Loop(26) = {48, -2, -47, 25};
//+
Plane Surface(26) = {26};
//+
Curve Loop(27) = {65, -28, -27, -26, -25};
//+
Plane Surface(27) = {27};
//+
Curve Loop(28) = {33, 64, -65, -24, -23};
//+
Plane Surface(28) = {28};
//+
Curve Loop(29) = {24, 47, -1, -45};
//+
Plane Surface(29) = {29};
//+
Curve Loop(30) = {23, 45, -7, -46};
//+
Plane Surface(30) = {30};
//+
Curve Loop(31) = {33, 52, -8, -46};
//+
Plane Surface(31) = {31};
//+
Curve Loop(32) = {38, 58, -17, -12, -56, 32};
//+
Plane Surface(32) = {32};
//+
Curve Loop(33) = {39, 40, 41, 42, 43, 44, 37, 63, 32, 38};
//+
Plane Surface(33) = {33};
//+
Curve Loop(34) = {42, 61, -20, 16, -19, -59, 40, 41};
//+
Plane Surface(34) = {34};
//+
Curve Loop(35) = {18, 19, -16, 20, 21, 22, -13, 62, 12, 17};
//+
Plane Surface(35) = {35};
//+
Curve Loop(36) = {44, 54, -22, -60};
//+
Plane Surface(36) = {36};


//+
Curve Loop(37) = {92, 93, -39, 58, 18, 94, -91, -90};
//+
Plane Surface(37) = {37};
//+
Curve Loop(38) = {95, 96, 97, -94, -59, -93};
//+
Plane Surface(38) = {38};
//+
Curve Loop(39) = {96, 99, 100, 98};
//+
Curve Loop(40) = {43, 60, -21, -61};
//+
Plane Surface(39) = {39, 40};
//+
Curve Loop(41) = {100, 101, 90, -102};
//+
Plane Surface(40) = {41};

// nektere dalsi pridane ploch
// 41 - stena tunelu s elektrodami
//+
Curve Loop(42) = {29, 56, -6, -51};
//+
Plane Surface(41) = {42};
//+
Curve Loop(43) = {34, 55, -9, -52};
//+
Plane Surface(42) = {43};
//+
Curve Loop(44) = {84, -83, -89, -15};
//+
Plane Surface(43) = {44};
//+
Curve Loop(45) = {101, 92, 95, -98};
//+
Plane Surface(44) = {45};
//+
Curve Loop(46) = {91, -97, 99, 102};
//+
Plane Surface(45) = {46};
//+
Curve Loop(47) = {63, 56, -62, -55};
//+
Plane Surface(46) = {47};
//+
Surface Loop(2) = {38, 44, 40, 39, 45, 37, 33, 34, 35, 36, 32, 18, 46};
//+
Curve Loop(48) = {4, -50, -27, 49};
//+
Plane Surface(47) = {48};
//+
Curve Loop(49) = {5, -66, 2, 3, 4};
//+
Plane Surface(48) = {49};
//+
Curve Loop(50) = {1, 66, 103, -8, 7};
//+
Plane Surface(49) = {50};



// body elektrod - ldp 
ldp_points[] = {};
xdist = 1;
xorigin = 0;
xdirection = -1;
yposition = 0;
zposition = 0.25;
dn = 201;
For i In {0 : 39}
  pn = dn + i; 
  Point(pn) = {xorigin + i * xdist * xdirection, yposition, zposition, len99 };
  ldp_points[] += {pn};
  Point{pn} In Surface{41};
EndFor

// body elektrod - lhp 
lhp_points[] = {};
xdist = 1;
xorigin = 0;
xdirection = -1;
yposition = 0;
zposition = 1.25;
dn = 301;
For i In {0 : 39}
  pn = dn + i; 
  Point(pn) = {xorigin + i * xdist * xdirection, yposition, zposition, len99 };
  lhp_points[] += {pn};
  Point{pn} In Surface{41};
EndFor

//Point{201, 202, 203, 204, 205} In Surface{41};
//Point{301, 302, 303, 304, 305} In Surface{41};

// objemy ?
//+
Surface Loop(101) = {45, 37, 44, 40, 39, 38, 34, 33, 36, 35, 32, 19, 43, 24, 2, 23, 13, 16, 49, 29, 28, 31, 30, 27, 47, 48, 26, 25, 11, 1, 9, 20, 17, 14, 15, 12, 21, 10};
//+
Volume(101) = {101};
Physical Volume(1) = {101};
//+
Surface Loop(102) = {13, 23, 24, 8, 43, 2, 16, 19, 12, 22, 41, 15, 11, 1, 9, 20, 17, 14, 7, 42, 18, 21, 10};
//+
Volume(102) = {102};
//+
Physical Volume(2) = {102};


//+
Physical Surface(1) = {30, 31, 29, 28, 49, 26, 25, 47, 27, 22, 48, 7, 8, 42, 41, 32, 18, 33, 35, 34, 36};
//+
Physical Surface(2) = {44, 38, 37, 45, 39, 40};

Physical Point(99) = ldp_points[];
//Physical Point(99) += lhp_points[];
