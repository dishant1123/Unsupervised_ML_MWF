"""
hierarchical clustering : that groups similar data points together into clusters building a tree like hierarchy and its called as dendrogram. it not  specify to require the  number of clustering in advance. 

types of hierarchical clustering:

1.agglomerative clustering :   botton  to  up  
    ex : 
    
    name     marks  
    A         20 
    B         25
    C         90 
    D         96
    
    separate cluster  : [A] , [B] , [C] , [D]  ===>start each point 
                        |
                        merge closest  points  : [A,B]   [C]  [D]
                        |
                        merge closest  points  : [A,B]  [C,D]
                        |
                        merge closest  points  : [A,B,C]  [D]
                        |
                        merge closest  points  : [A,B,C,D]

                ABCD 
                  |
               [a]  [b]  [c]  [d]
                | 
               [a,b]  [c] [d]
                |
                [a,b]  [c,d]
                | 
                [a,b,c]  [d]
                |
                [a,b,c,d] 

2.divisive clustering : up  to  bottom
    ex : 
    
    name     marks  
    A         20 
    B         25
    C         90 
    D         96
    
    [A,B,C,D]
        |
    [A,B,C]  [D]
        |
    [A,B]  [C,D]
        |
    [A]  [B,C,D]
        |
    [A]  [B] [C]  [D]

linkage  : 
1.single  linkage :  the  distance  between  two  clusters  is  the  average  of  the  distances  between  each  pair  of  points  in  the  two  clusters.

2. complete linkage :  the  distance  between  two  clusters  is  the  maximum  of  the  distances  between  each  pair  of  points  in  the  two  clusters.

3. average linkage :  the  distance  between  two  clusters  is  the  average  of  the  distances  between  each  pair  of  points  in  the  two  clusters.

4. Ward's linkage :  the  distance  between  two  clusters  is  the  sum  of  the  squares  of  the  differences  of  each  pair  of  points  in  the  two  clusters.
"""

