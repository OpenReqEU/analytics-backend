import re


def regex_tags_extractor(text):
    IP_address = list(set(re.findall(pattern=r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', string=text)))
    AS_code = list(set(re.findall(pattern=r'AS\d+', string=text)))
    CVE_code = list(set(re.findall(pattern=r'CVE-\d+-\d+', string=text)))
    SHA1_code = list(set(re.findall(pattern=r'\b[0-9a-f]{40}\b', string=text)))
    #  Imme Emosol URL regex
    IE_url_regex = r"(?:(?:https?|ftp)://)(?:\S+(?::\S*)?@)?(?:(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\u00a1-\uffff0-9]+-?)*[a-z\u00a1-\uffff0-9]+)(?:\.(?:[a-z\u00a1-\uffff0-9]+-?)*[a-z\u00a1-\uffff0-9]+)*(?:\.(?:[a-z\u00a1-\uffff]{2,})))(?::\d{2,5})?(?:/[^\s]*)?"
    URLs = list(set(re.findall(pattern=IE_url_regex, string=text)))
    File_Extension = list(set(re.findall(pattern=r'(\b\w+\.[a-z]{2,4}\b)', string=text)))
    Files = list(set(re.findall(pattern=r'\w+File\b', string=text)))
    IEC = list(set(re.findall(pattern=r'\bIEC\s\d{3,5}\-*\d*\-*\d*\b', string=text)))
    Port = list(set(re.findall(pattern=r'port\s\d{1,}\b', string=text)))
    sigle = list(set([x[0] for x in re.findall(pattern=r'\b([A-z]+(-\d+)+)\b', string=text)]))
    regex_data = r'\b((Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s(\d{1,2}(rd|th|st|nd)?\s\d{4}))\b'
    date = list(set([x[0] for x in re.findall(pattern=regex_data, string=text)]))
    data_bs = list(set(re.findall(pattern=r'(\d+/\d+/\d+)', string=text)))
    data_dash = list(set(re.findall(pattern=r'(\d{4}\-\d{2}\-\d{2})', string=text)))

    regex_list = IP_address + AS_code + CVE_code + SHA1_code + URLs + File_Extension + Files + IEC + Port + sigle + date + data_bs + data_dash

    regex_dict = {'IP_address': IP_address, 'AS_code': AS_code, 'CVE_code': CVE_code,
                  'SHA1_code': SHA1_code, 'URLs': URLs, 'File_Extension': File_Extension,
                  'Files': Files, 'IEC': IEC, 'Port': Port, 'sigle': sigle, 'date': date,
                  'data_bs': data_bs, 'data_dash': data_dash}

    tags_list = []
    for item in regex_list:
        item2 = item.replace(" ", "_")
        text = text.replace(item, item2)
        tags_list.append(item2)

    return tags_list, text