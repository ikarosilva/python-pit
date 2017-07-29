'''
Created on Jul 24, 2017

@author: isilva
'''


def run():
    years=['2015','2016','2017','2018']
    months=['July','Aug','Sept','Oct','Nov','Dec','Jan','Feb','March','April','May','June']
    rate=1000
    count=0
    for year in years:
        for month in months:
            count=count+rate
            print year + ' ' + month  +' :' + str(count)
            

if __name__ == "__main__":
    run()