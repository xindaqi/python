import random
import os
def process_cut(source_path, cut_path):
	'''Process source data.
	Save data expect E,M.

	:params source_path: source dialog path
	:params cut_path: save dialog path

	return:
	convs: complete dialogs.
	'''

	'''Saved all conversations.'''
	convs = []
	with open(source_path, 'r', encoding='utf8') as f:
		'''<_io.TextIOWrapper name='./data/source_data.conv' mode='r' encoding='utf8'>'''
		print("open context object: {}".format(f))
		# data = f.readlines()
		'''['E\n', 'M 呵呵\n', 'M 是王若猫的。\n]'''
		# print("data: {}".format(data))
		# one_conv = []
		'''Complete dialog: contains Question and Answer.'''
		complete_dialog = []
		for line in f:
			'''Delete line feed symbol: \n'''
			line = line.strip('\n')
			
			if line == "":
				continue
			if line[0] == "E":
				if complete_dialog:
					'''Add dialog to conversations list.'''
					convs.append(complete_dialog)
					complete_dialog = []
			if line[0] == 'M':
				'''Extract Question and Answer which contains in M'''
				complete_dialog.append(line[1:])
				'''
				contain M: M 三鹿奶粉也假，不一样的卖啊

				'''
				# print("contain M: {}".format(line))
			'''
			line data: E
			
			line data: M 呵呵
			
			'''
			# print("line data: {}".format(line))
		# print("Complete dialog {}".format(complete_dialog))
	# print("All complete dialog: {}".format(convs))
	return convs
def question_answer(convs):
	'''Extract questions and answers from dialog.
	:params convs: dialogs.

	return:
	questions: questions
	answers: answers
	'''
	questions = []
	answers = []
	for conv in convs:
		if len(conv) == 1:
			continue
		if len(conv) % 2 != 0:
			'''if dialog was not one to one, delete the last one and keep Q&A.'''
			conv = conv[:-1]
		for i in range(len(conv)):
			'''Extract Question.'''
			if i % 2 == 0:
				questions.append(conv[i])
			else:
				'''Extract Answer.'''
				answers.append(conv[i])
	print("questions: {} \n answers: {}".format(questions, answers))
	return questions, answers


def save_question_answer(questions, answers, test_size,
							train_question_path, train_answer_path,
							test_question_path, test_answer_path):
	'''Save question and answer dataset.
	:params questions: question
	:params answers: answer
	:params test_size: set test data number and save
	:params train_question_path: question dataset path for train 
	:params train_answer_path: answer dataset path for train
	:params test_question_path: question dataset path for test
	:params test_answer_path: answer dataset path for test
	'''
	'''Train dataset.'''
	train_quesition_enc = open(train_question_path, "w")
	train_answer_dec = open(train_answer_path, "w")
	'''Test dataset.'''
	test_question_enc = open(test_question_path, "w")
	test_answer_dec = open(test_answer_path, "w")
	'''Random get test dateset which number is test_size.''' 
	test_index = random.sample([i for i in range(len(questions))], test_size)

	for i in range(len(questions)):
		if i in test_index:
			test_question_enc.write(questions[i]+'\n')
			test_answer_dec.write(answers[i]+'\n')
		else:
			train_quesition_enc.write(questions[i]+'\n')
			train_answer_dec.write(answers[i]+'\n')
	train_quesition_enc.close()
	train_answer_dec.close()
	test_question_enc.close()
	test_answer_dec.close()


def save_test():
	'''Read and save dataset.'''
	source_path = "./data/source_data.conv"
	convs = process_cut(source_path, None)
	print("convs: {}".format(convs))
	questions, answers = question_answer(convs)
	# print("questions: {} \n answers: {}".format(questions, answers))
	folder_list = ["./data/train/", "./data/test/"]
	file_list = ["./data/train/question.enc", "./data/train/answer.dec", "./data/test/question.enc", "./data/test/answer.dec"]
	for i in range(len(folder_list)):
		if not os.path.exists(folder_list[i]):
			os.makedirs(folder_list[i])
	for i in range(len(file_list)):
		if not os.path.exists(file_list[i]):
			os.mknod(file_list[i])
	'''Seting train dataset path.'''
	train_question_path = file_list[0]
	train_answer_path = file_list[1]
	'''Seting test dataset path.'''
	test_question_path = file_list[2]
	test_answer_path = file_list[3]
	save_question_answer(questions, answers, 5,
						train_question_path, train_answer_path,
						test_question_path, test_answer_path)

def generate_vocabulary(datasets, vocabulary_data):
	PAD = "__PAD__"
	GO = "__GO__"
	EOS = "__EOS__"  # 对话结束
	UNK = "__UNK__"  # 标记未出现在词汇表中的字符
	START_VOCABULART = [PAD, GO, EOS, UNK]
	PAD_ID = 0
	GO_ID = 1
	EOS_ID = 2
	UNK_ID = 3
	file_list = ["./data/train/question.enc", "./data/train/answer.dec", "./data/test/question.enc", "./data/test/answer.dec"]
	vocabulary = {}
	new_vocabulary = []
	with open(datasets, "r") as f:
		counter = 0
		for line in f:
			counter += 1
			'''Delete lind feed symbol: \n, and extract word in sentence.'''
			tokens = [word for word in line.strip()]
			for word in tokens:
				if word in vocabulary:
					vocabulary[word] += 1
				else:
					vocabulary[word] = 1

	vocabulary_list = START_VOCABULART + sorted(vocabulary, key=vocabulary.get, reverse=True)
	print("vocabulary: {}".format(vocabulary_list))
	with open(vocabulary_data, "w") as f:
		for word in vocabulary_list:
			f.write(word+'\n')

def generate_vocabulary_test():
	file_list = ["./data/train/question.enc", "./data/train/answer.dec", "./data/test/question.enc", "./data/test/answer.dec"]
	voc_list = ["./data/train/question_voc", "./data/train/answer_voc"]
	for i in range(len(voc_list)):
		if not os.path.exists(voc_list[i]):
			os.mknod(voc_list[i])
		generate_vocabulary(file_list[i], voc_list[i])

def word_to_vector(dataset_qa, vocabulary, vector):
	UNK_ID = 3
	tmp_vocab = []
	with open(vocabulary, "r") as f:
		'''Append word one by one to list as dependent element not entirely append to list.'''
		tmp_vocab.extend(f.readlines())
	'''Delete line feed: \n'''
	tmp_vocab = [line.strip() for line in tmp_vocab]
	'''Trans tmp_vocab to this format[()] and then convert to dict by dict{key:value}.'''
	vocab = dict([(x,y) for (y,x) in enumerate(tmp_vocab)])
	'''vocabulay dictionary: {'__PAD__': 0, '__GO__': 1, '__EOS__': 2, '__UNK__': 3, '是': 4, '谁': 5, '许': 6, '兵': 7, '呵': 8, '么': 9, '不': 10, '短': 11, '信': 12, '你': 13, '知': 14, '道': 15, '这': 16, '假': 17, '傻': 18, '逼': 19}
'''
	print("vocabulay dictionary: {}".format(vocab))
	with open(vector, "w") as f_vector:
		with open(dataset_qa, "r") as f_qa:
			for line in f_qa:
				line_vec = []
				for words in line.strip():
					line_vec.append(vocab.get(words, UNK_ID))
				# print("line vector: {}".format(line_vec))
				f_vector.write(" ".join([str(num) for num in line_vec]) + '\n')


def process_data(dataset_qa, vocabulary, vector):
	'''Read and save dataset.'''
	source_path = "./data/source_data.conv"
	convs = process_cut(source_path, None)
	print("convs: {}".format(convs))
	questions, answers = question_answer(convs)
	# print("questions: {} \n answers: {}".format(questions, answers))
	folder_list = ["./data/train/", "./data/test/"]
	file_list = ["./data/train/question.enc", "./data/train/answer.dec", "./data/test/question.enc", "./data/test/answer.dec"]
	for i in range(len(folder_list)):
		if not os.path.exists(folder_list[i]):
			os.makedirs(folder_list[i])
	for i in range(len(file_list)):
		if not os.path.exists(file_list[i]):
			os.mknod(file_list[i])
	'''Seting train dataset path.'''
	train_question_path = file_list[0]
	train_answer_path = file_list[1]
	'''Seting test dataset path.'''
	test_question_path = file_list[2]
	test_answer_path = file_list[3]
	save_question_answer(questions, answers, 5,
						train_question_path, train_answer_path,
						test_question_path, test_answer_path)
	PAD = "__PAD__"
	GO = "__GO__"
	EOS = "__EOS__"  # 对话结束
	UNK = "__UNK__"  # 标记未出现在词汇表中的字符
	START_VOCABULART = [PAD, GO, EOS, UNK]
	PAD_ID = 0
	GO_ID = 1
	EOS_ID = 2
	UNK_ID = 3
	voc_list = ["./data/train/question_voc", "./data/train/answer_voc"]
	for i in range(len(voc_list)):
		if not os.path.exists(voc_list[i]):
			os.mknod(voc_list[i])
		generate_vocabulary(file_list[i], voc_list[i])
	word_to_vector(dataset_qa, vocabulary, vector)












if __name__ == "__main__":
	process_data("./data/train/question.enc","./data/train/question_voc", "./data/train/question.vec")
	


