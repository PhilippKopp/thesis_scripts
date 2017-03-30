import sys
import numpy as np
import math



########## Variables 
# indices of vertices of certain points (seen from face):
# nosetip, right human eye outer, right human eye inner, left human eye inner, left human eye outer, nose middle down, mouth top, mouth right, mouth bottom, mouth left

# surrey registered model
surrey_eye_vertices = [171, 604] #right eye centre, left eye centre
surrey_outer_eye_vertices = [177, 610] # right eye outer, left eye outer
#surrey_imp_vertices = [114, 177, 181, 614, 617, 270, 424, 398, 401, 812] old and left inner eye is wrong!!!!
surrey_imp_vertices = [114, 177, 181, 614, 610, 270, 424, 398, 401, 812]

mask = [0, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 292, 295, 296, 297, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 342, 344, 345, 346, 347, 348, 351, 353, 354, 374, 375, 376, 377, 378, 379, 387, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 442, 443, 444, 447, 448, 449, 450, 451, 452, 453, 455, 456, 457, 458, 459, 460, 461, 462, 463, 465, 467, 468, 469, 470, 471, 472, 473, 474, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 717, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 759, 761, 762, 763, 764, 765, 768, 770, 771, 791, 792, 793, 794, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 851, 852, 853, 854, 855, 856, 859, 860, 865, 866, 867, 868, 871, 872, 877, 878, 879, 880, 883, 884, 887, 888, 891, 892, 893, 894, 897, 898, 899, 900, 903, 904, 911, 912, 917, 918, 919, 920, 937, 938, 939, 940, 945, 946, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 969, 970, 977, 978, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 993, 994, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1045, 1046, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1059, 1060, 1061, 1062, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1145, 1146, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1179, 1180, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1640, 1641, 1642, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1698, 1699, 1700, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1743, 1747, 1748, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1774, 1775, 1776, 1777, 1778, 1779, 1781, 1784, 1796, 1797, 1799, 1800, 1802, 1803, 1804, 1805, 1806, 1807, 1816, 1817, 1818, 1820, 1821, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1860, 1861, 1868, 1869, 1870, 1871, 1872, 1873, 1880, 1881, 1884, 1887, 1892, 1893, 1894, 1895, 1896, 1897, 1907, 1908, 1910, 1911, 1913, 1921, 1922, 1924, 1925, 1926, 1927, 1929, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1955, 1957, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2008, 2009, 2011, 2012, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2028, 2029, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2067, 2068, 2069, 2070, 2071, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2123, 2124, 2125, 2126, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2156, 2159, 2160, 2162, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2227, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254, 2255, 2256, 2259, 2260, 2261, 2262, 2263, 2264, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2326, 2327, 2328, 2333, 2334, 2335, 2336, 2337, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2394, 2395, 2396, 2397, 2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2424, 2425, 2426, 2427, 2428, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2481, 2482, 2483, 2484, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2505, 2506, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2536, 2537, 2538, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2573, 2574, 2575, 2576, 2577, 2582, 2583, 2584, 2585, 2586, 2587, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2644, 2645, 2646, 2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688, 2689, 2690, 2691, 2692, 2693, 2694, 2695, 2696, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720, 2721, 2722, 2723, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740, 2741, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2770, 2771, 2772, 2773, 2774, 2775, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811, 2812, 2813, 2814, 2815, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826, 2827, 2828, 2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2870, 2871, 2872, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889, 2890, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2927, 2928, 2929, 2930, 2931, 2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3048, 3049, 3050, 3051, 3052, 3053, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150, 3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3342, 3343, 3344, 3345, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, 3359, 3360, 3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374, 3375, 3376, 3377, 3378, 3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447]


ALL_POINTS = "ALL_POINTS"
	


def get_KF_ITW_vertex_ids(ID, EXPRESSION):
	vertices = None
	# Imperial KF-ITW Ground Truth
	if (ID =='/02/'):
		if (EXPRESSION=='/neutral/'):
			vertices = [236459, 175007, 176204, 178225, 178720, 253473, 287583, 298637, 303777, 290451]  # KF-ITW GT 02 neutral
		elif (EXPRESSION=='/happy/'):
			vertices = [227198, 164924, 170017, 175546, 197434, 250220, 291404, 294816, 291809, 295019]  # KF-ITW GT 02 happy
		elif (EXPRESSION=='/surprised/'):
			vertices = [225788, 167612, 168465, 172740, 174774, 247124, 291409, 305141, 329892, 311767]  # KF-ITW GT 02 surprised
	elif (ID =='/08/'):
		if (EXPRESSION=='/neutral/'):
			vertices = [178641, 146718, 144118, 144962, 144724, 220304, 257470, 259248, 257325, 262390]  # KF-ITW GT 08 neutral
		elif (EXPRESSION=='/surprised/'):
			vertices = [181593, 146029, 145197, 150569, 172910, 226319, 260380, 277372, 319772, 277468]  # KF-ITW GT 08 surprised
	elif (ID =='/11/'):
		if (EXPRESSION=='/neutral/'):
			vertices = [207882, 155735, 155091, 158820, 163516, 230091, 267013, 262134, 267173, 269503]  # KF-ITW GT 11 neutral
		elif (EXPRESSION=='/happy/'):
			vertices = [185150, 152127, 129621, 127992, 147955, 204231, 223863, 246432, 247629, 244016]  # KF-ITW GT 11 happy
		elif (EXPRESSION=='/surprised/'):
			vertices = [177168, 145275, 120284, 124374, 118427, 221424, 238129, 261843, 279287, 261413]  # KF-ITW GT 11 surprised
	elif (ID =='/13/'):
		if (EXPRESSION=='/neutral/'):
			vertices = [174709, 142579, 142040, 128189, 128981, 196463, 238786, 249384, 239378, 240807]  # KF-ITW GT 13 neutral
		elif (EXPRESSION=='/happy/'):
			vertices = [194111, 138123, 139067, 139235, 143847, 211242, 228953, 247955, 243910, 251262]  # KF-ITW GT 13 happy
		elif (EXPRESSION=='/surprised/'):
			vertices = [188179, 136744, 136328, 135283, 137054, 206912, 224366, 259348, 285635, 260112]  # KF-ITW GT 13 surprised
	elif (ID =='/16/'):
		if (EXPRESSION=='/neutral/'):
			vertices = [176754, 139940, 143421, 146310, 147505, 195249, 225684, 226766, 225761, 229664]  # KF-ITW GT 16 neutral
		elif (EXPRESSION=='/happy/'):
			vertices = [220826, 184567, 185797, 171628, 191477, 239258, 254915, 270174, 267829, 272240]  # KF-ITW GT 16 happy

	return vertices


class OalException(Exception):
	pass

def get_vertex_positions(obj_file, imp_vertices):
	"""
	opens a obj file and searches for the imp_vertices given as indices
	returns a numpy matrix with the coordinates of all the imp_vertices
	"""
	
	mesh = read_mesh(obj_file)

	if imp_vertices == ALL_POINTS:
		imp_vertices = [x for x in range(len(mesh))]
	imp_coordinates = np.empty((len(imp_vertices),3), dtype=float)

	for coor_index, imp_vertex in enumerate(imp_vertices):
		imp_coordinates[coor_index, :] = mesh[imp_vertex]

	return imp_coordinates						

def write_aligned_obj (input_obj, tranformation_params, output_obj):
	"""
	Takes an input obj file and transformation params as dictionary like tform = {'rotation':T, 'scale':b, 'translation':c}
	Then writes to the outputfile with the new aligned obj
	"""
		
	T = tranformation_params['rotation']
	b = tranformation_params['scale']
	c = tranformation_params['translation']

	with open(input_obj, "r") as imperial_obj:
		with open(output_obj, "w") as surrey_obj:
			for line in imperial_obj:
		
				if (line.startswith('v ')):
					coordinates = [float(i) for i in line.split()[1:]]
					coordinates = np.array(coordinates)
					new_coordinates = np.dot(b,np.dot(coordinates,T)) + c
					line_out = 'v'
					for i in new_coordinates:
						line_out= line_out + ' ' + str(i)
					line_out+='\n'
				else:
					line_out = line
				surrey_obj.write(line_out)


def read_mesh(obj_file):
	""" small helper function that loads a obj and returns a mesh as list of coordinates"""
	mesh =[]
	with open(obj_file, "r") as obj:

		# header line first
		header = obj.readline()
		if header.startswith('v '): #if no header jump back to beginning
			obj.seek(0)

		for line in obj:
			if (line.startswith('v ')):
				coordinates = [float(i) for i in line.split()[1:]]
				mesh.append(coordinates)
			else:
				continue
	return mesh

def write_mesh(mesh, traingle_list_from_file, output_obj):
	with open(output_obj, "w") as out:

		# first write vertex positions from mesh
		for vertex in mesh:
			line_out = 'v'
			for coordinate in vertex:
				line_out= line_out + ' ' + str(coordinate)
			line_out+='\n'
			out.write(line_out)

		# then write triangel list from other reference file
		with open(traingle_list_from_file, "r") as reference:
			for line in reference:
				if (line.startswith('vt ') or line.startswith('f ')):
					out.write(line)


def calc_distance (a, b):
	if (not len(a)==len(b)):
		raise OalException('Can\'t calculate distance between points with different dimensions!')
	a_np = np.array(a)
	b_np = np.array(b)
	return np.linalg.norm(a_np-b_np, ord=2)
	# return np.spatial.distance.euclidean(a,b)


def measure_distances_non_registered(fitted_obj_file, aligned_gt_obj_file, measure_on_fitted_vertices=ALL_POINTS):
	"""
	takes a fitted obj file and an aligned gt obj file, between them the distance gets measured
	at the vertices specified in measure_on_fitted_vertices
	returns a list of distances and the vertices in the gt obj that have shortest distance
	"""
	distances =[]
	corresponding_vertices_gt =[]
	fitted_mesh = read_mesh(fitted_obj_file)
	gt_mesh = read_mesh(aligned_gt_obj_file)

	for index_fitted in range(len(fitted_mesh)):
		# if index in list of given vertices or if list empty measure all distances
		if (index_fitted in measure_on_fitted_vertices or measure_on_fitted_vertices==ALL_POINTS):

			shortest_distance = 100000000
			index_shortest = -1
			# go through entire gt mesh and find vertex with smallest distance
			for index_gt in range(len(gt_mesh)):
				distance = calc_distance(fitted_mesh[index_fitted], gt_mesh[index_gt])
				if distance< shortest_distance:
					shortest_distance = distance
					index_shortest = index_gt
			corresponding_vertices_gt.append(index_shortest)
			distances.append(shortest_distance)
			#print "for vertex "+str(index_fitted)+ " (fitted) the nearest index in gt is "+str(index_shortest)+" with a distance of "+str(shortest_distance)
	return distances, corresponding_vertices_gt

def measure_distances_registered(fitted_obj_file, aligned_gt_obj_file, mask=ALL_POINTS):
	"""
	takes two registered obj models
	returns a list of distances
	"""
	fitted_mesh = read_mesh(fitted_obj_file)
	gt_mesh = read_mesh(aligned_gt_obj_file)
	fitted_mesh_np = np.array(fitted_mesh)
	gt_mesh_np = np.array(gt_mesh)
	diff = np.linalg.norm(fitted_mesh_np-gt_mesh_np, ord=2, axis=1)
	if mask!=ALL_POINTS:
		diff_mask =[]
		for index in range(len(diff)):
			if index in mask:
				diff_mask.append(diff[index])
		return diff_mask

	else:
		return diff 

def pseudocolor(val, minval, maxval):
	# from here: http://stackoverflow.com/questions/10901085/range-values-to-pseudocolor
	import colorsys

	# convert val in range minval..maxval to the range 0..120 degrees which
	# correspond to the colors red..green in the HSV colorspace
	h = (float(val-minval) / (maxval-minval)) * 120
	h=120-h
	# convert hsv color (h,1,1) to its rgb equivalent
	# note: the hsv_to_rgb() function expects h to be in the range 0..1 not 0..360
	r, g, b = colorsys.hsv_to_rgb(h/360, 1., 1.)
	return r, g, b

def write_error_mesh_registered(fitted_obj_file, aligned_gt_obj_file, error_mesh):
	"""
	takes two registered obj models and a path to an output model
	writes mesh with color coding of errors
	"""
	fitted_mesh = read_mesh(fitted_obj_file)
	gt_mesh = read_mesh(aligned_gt_obj_file)
	fitted_mesh_np = np.array(fitted_mesh)
	gt_mesh_np = np.array(gt_mesh)
	distances = np.linalg.norm(fitted_mesh_np-gt_mesh_np, ord=2, axis=1)
	max_error = max(distances)
	colors = [pseudocolor(x, 0, max_error) for x in distances]

	#print (colors)
	with open(error_mesh, "w") as out:

		# first write vertex positions from mesh
		for vertex_id in range(len(fitted_mesh)):
			line_out = 'v'
			for coordinate in fitted_mesh[vertex_id]:
				line_out= line_out + ' ' + str(coordinate)
			for color in colors[vertex_id]:
				line_out= line_out + ' ' + str(color)
			line_out+='\n'
	
			out.write(line_out)
	
		# then write triangel list from other reference file
		with open(aligned_gt_obj_file, "r") as reference:
			for line in reference:
				if (line.startswith('f ')):
					out.write(line)
	



def menpo3d_non_rigid_icp (fitted_obj, gt_obj, fitted_imp_3d_points, gt_imp_3d_points, output_obj):
	import sys
	#sys.path.append("/user/HS204/m09113/scripts/menpo_playground/src/lib/python3.5/site-packages")
	#sys.path.append("/user/HS204/m09113/miniconda2/envs/menpo/lib/python2.7/site-packages/")
	from menpo3d.correspond import non_rigid_icp
	from menpo3d.io.output.base import export_mesh
	import menpo3d.io as m3io
	import menpo


	# try something
	# lm_weights = [5, 2, .5, 0, 0, 0, 0, 0]  # default weights
	# lm_weights = [10, 8, 5, 3, 2, 0.5, 0, 0]
	lm_weights = [25, 20, 15, 10, 8, 5, 3, 1]
	# lm_weights = [2, 1, 0, 0, 0, 0, 0, 0]
	# lm_weights = [25, 20, 15, 10, 5, 2, 1, 0]
	# lm_weights = [100, 0, 0, 0, 0, 0, 0, 0]
	# lm_weights = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
	
	stiff_weights = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2]  # default weights
	# stiff_weights = [50, 20, 15, 10, 3, 1, 0.35, 0.2]
	# stiff_weights = [50, 40, 30, 20, 10, 8, 5, 2]
	# stiff_weights = [50, 20, 10, 5, 2, 1, 0.5, 0.2]
	# stiff_weights = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
	
	# load pointcloud surrey model as src
	src = m3io.import_mesh(fitted_obj)
	
	# load scan mesh as dest
	dest = m3io.import_mesh(gt_obj)
	#print('destination loaded')
	
	# add landmarks to mesh
	src.landmarks['myLM'] = menpo.shape.PointCloud(fitted_imp_3d_points)
	dest.landmarks['myLM'] = menpo.shape.PointCloud(gt_imp_3d_points)
	#print('landmarks loaded')
	
	# non rigid icp pointcloud as result
	#marc org
	result = non_rigid_icp(src, dest, eps=1e-3, landmark_group='myLM', stiffness_weights=stiff_weights, data_weights=None,
					   landmark_weights=lm_weights, generate_instances=False, verbose=False)
	
	# export the result mesh
	export_mesh(result, output_obj, extension='.obj', overwrite=True)
	


# might be interesting:
#https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.spatial.procrustes.html

#this code stolen from here: http://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
def procrustes(X, Y, scaling=True, reflection='best'):
	"""
	A port of MATLAB's `procrustes` function to Numpy.

	Procrustes analysis determines a linear transformation (translation,
	reflection, orthogonal rotation and scaling) of the points in Y to best
	conform them to the points in matrix X, using the sum of squared errors
	as the goodness of fit criterion.

		d, Z, [tform] = procrustes(X, Y)

	c - Translation component
	T - Orthogonal rotation and reflection component
	b - Scale component

	Z = b*Y*T + c;

	Inputs:
	------------
	X, Y    
		matrices of target and input coordinates. they must have equal
		numbers of  points (rows), but Y may have fewer dimensions
		(columns) than X.

	scaling 
		if False, the scaling component of the transformation is forced
		to 1

	reflection
		if 'best' (default), the transformation solution may or may not
		include a reflection component, depending on which fits the data
		best. setting reflection to True or False forces a solution with
		reflection or no reflection respectively.

	Outputs
	------------
	d       
		the residual sum of squared errors, normalized according to a
		measure of the scale of X, ((X - X.mean(0))**2).sum()

	Z
		the matrix of transformed Y-values

	tform   
		a dict specifying the rotation, translation and scaling that
		maps X --> Y

	"""

	n,m = X.shape
	ny,my = Y.shape

	muX = X.mean(0)
	muY = Y.mean(0)

	X0 = X - muX
	Y0 = Y - muY

	ssX = (X0**2.).sum()
	ssY = (Y0**2.).sum()

	# centred Frobenius norm
	normX = np.sqrt(ssX)
	normY = np.sqrt(ssY)

	# scale to equal (unit) norm
	X0 /= normX
	Y0 /= normY

	if my < m:
		Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

	# optimum rotation matrix of Y
	A = np.dot(X0.T, Y0)
	U,s,Vt = np.linalg.svd(A,full_matrices=False)
	V = Vt.T
	T = np.dot(V, U.T)

	if reflection is not 'best':

		# does the current solution use a reflection?
		have_reflection = np.linalg.det(T) < 0

		# if that's not what was specified, force another reflection
		if reflection != have_reflection:
			V[:,-1] *= -1
			s[-1] *= -1
			T = np.dot(V, U.T)

	traceTA = s.sum()

	if scaling:

		# optimum scaling of Y
		b = traceTA * normX / normY

		# standarised distance between X and b*Y*T + c
		d = 1 - traceTA**2

		# transformed coords
		Z = normX*traceTA*np.dot(Y0, T) + muX

	else:
		b = 1
		d = 1 + ssY/ssX - 2 * traceTA * normY / normX
		Z = normY*np.dot(Y0, T) + muX

	# transformation matrix
	if my < m:
		T = T[:my,:]
	c = muX - b*np.dot(muY, T)

	#transformation values 
	tform = {'rotation':T, 'scale':b, 'translation':c}

	return d, Z, tform


def write_colored_mesh(obj_file, mask, outputfile):
	"""
	
	"""
	mesh = read_mesh(obj_file)
	color_mask = pseudocolor(0.1, 0, 1)
	color_rest = pseudocolor(0.9, 0, 1)

	triangle_list = []
	with open(obj_file, "r") as reference:
			for line in reference:
				if (line.startswith('f ')):
					triangle_list.append(line)


	#print (colors)
	with open(outputfile, "w") as out:

		# first write vertex positions from mesh
		for vertex_id in range(len(mesh)):
			line_out = 'v'
			for coordinate in mesh[vertex_id]:
				line_out= line_out + ' ' + str(coordinate)
			if vertex_id in mask:
				for rgb in color_mask:
					line_out= line_out + ' ' + str(rgb)
			else:
				for rgb in color_rest:
					line_out= line_out + ' ' + str(rgb)
			line_out+='\n'
	
			out.write(line_out)
	
		for triangel in triangle_list:
			out.write(triangel)
