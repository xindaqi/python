import scrapy
from w3lib.html import remove_tags

class BaiduSearchSpider(scrapy.Spider):
	name = "patent_search"
	# allowed_domains = ["baidu.com"]
	allowed_domains = ["soopat.com"]
	start_urls = [
		# "https://www.baidu.com/s?wd=自然语言处理"
		"http://www2.soopat.com/Home/Result?SearchWord=图像风格"
		# "http://www2.soopat.com/Home/Result?SearchWord=预训练"
	]
	# group = 0
	# count = 0

	
	def parse(self, response):
		# group += 1
		# count += 1
		# file_name = "baidu.html"
		# hrefs = response.selector.xpath('//div[@class="result c-container "]/h3/a/@href').extract()
		# hrefs = response.xpath('//div[contains(@class, "c-container")]/h3/a/@href').extract()

		# title = response.xpath('//div[contains(@class, "PatentBlock")]/div/h2/font/input/@mc').extract()
		# print("hrefs: {}".format(len(hrefs)))
		# title_num = len(title)
		# count += title_num
		'''output result info'''
		# yield {
		# 	"title":title
		# }

		# print("Title group{}: {}".format(title_num, title))
		# print("Title number: {}".format(title_num))

		
		containers = response.selector.xpath('//div[contains(@class, "PatentBlock")]')
		print("Containers: {}".format(containers))
		for container in containers:
			href = container.xpath('div/h2/a/@href').extract()[0]
			# href = container.xpath('a/@href').extract()
			print("href: {}".format(href))
			title = container.xpath('div/h2/font/input/@mc').extract()[0]
			# title = container.xpath('font/input/@mc').extract()
			print("Title: {}".format(title))
		# 	title = remove_tags(container.xpath('h3/a').extract()[0])
			
		# 	c_abstract = container.xpath('div/div/div[contains(@class, "c-abstract")]').extract()
		# 	abstract = ""
		# 	if len(c_abstract) > 0:
		# 		# abstract = c_abstract[0]
		# 		abstract = remove_tags(c_abstract[0])

			request = scrapy.Request(href, callback=self.parse_url)


		# 	request.meta['title'] = title
		# 	request.meta['abstract'] = abstract

			yield request


		# next_page = response.xpath('//div[@id="page"]/a/@href').extract()

		next_page = response.xpath('//div[@id="SoopatPager"]/a/@href').extract()
		print("Page href: {}".format(next_page))
		next_page = next_page[-1]
		if next_page is not None:
			yield response.follow(next_page, self.parse)



	def parse_url(self, response):
	# 	print("URL: {}".format(response.url))
	# 	print("Title: {}".format(response.meta['title']))
	# 	print("Abstract: {}".format(response.meta['abstract']))
		content = remove_tags(response.xpath('//body/div[@class="lay"]/div/div/div/table/tbody/tr/td/text()').extract()[0])
	# 	print("Content length: {}".format(len(content)))
		print("Content: {}".format(content))

		# next_pages = response.css('a.n a::attr(href)').get()
		# next_page = response.xpath('//a[@class="n"]/@href').extract()
		# next_page = response.xpath('//div[@id="page"]/a/@href').extract()
		# next_page = next_page[-1]

		# # print("Pages nums: {}".format(len(next_pages)))
		# print("Pages: {}".format(next_page))

		# if next_page is not None:

		# 	yield response.follow(next_page, self.parse)


			
		# if next_page is not None and self.i < 5:
		# 	print("turns: {}".format(self.i))
		# 	self.i += 1
		# 	yield response.follow(next_page, self.parse)

		# for href in hrefs:
		# 	print("href: {}".format(href))
			# print(href)

		# with open(file_name, 'wb') as f:
		# 	f.write(response.body)
		# print("Response body: {}".format(response.body))






# class QuotesSpider(scrapy.Spider):
# 	name = "quotes"
# 	start_urls = [
# 		'http://quotes.toscrape.com/tag/humor',
# 	]

# 	def parse(self, response):
# 		for quote in response.css('div.quote'):
# 			yield {
# 				'text': quote.css('span.text::text').get(),
# 				'author': quote.xpath('span/small/text()').get(),
# 			}

# 		next_page = response.css('li.next a::attr("href")').get()
# 		if next_page is not None:
# 			yield response.follow(next_page, self.parse)