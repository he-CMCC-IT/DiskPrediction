from suds.client import Client
from suds.xsd.doctor import ImportDoctor,Import
import datetime
print(datetime.datetime.now())
url= "http://www.webxml.com.cn/WebServices/WeatherWebService.asmx?wsdl"
imp= Import('http://www.w3.org/2001/XMLSchema', location='http://www.w3.org/2001/XMLSchema.xsd')
imp.filter.add("http://WebXml.com.cn/")
d= ImportDoctor(imp)
client= Client(url,doctor=d)
result= client.service.getWeatherbyCityName("杭州")
print(result)
print(datetime.datetime.now())

'''
第二个实例
'''
# imp= Import('http://www.w3.org/2001/XMLSchema', location='http://www.w3.org/2001/XMLSchema.xsd')#导入
# imp.filter.add("http://WebXml.com.cn/")#<em>namespace如下图</em>
# d= ImportDoctor(imp)
# url = "http://fy.webxml.com.cn/webservices/EnglishChinese.asmx?wsdl"
# client= Client(url,doctor=d)
# result = client.service.TranslatorReferString("字母")
# print(result)
