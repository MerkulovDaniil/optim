# USAGE:
# python create_collection.py --title "Seminar" --chapters "Matrix calculus", "Gradient descent", "#ex#Matrix calculus#1,2,3"
# Parsing arguments lib
import argparse 
import os, shutil 
 
parser = argparse.ArgumentParser() 
parser.add_argument( 
    '--title', 
    default='Seminar', 
    help='provide the name of the collection (string)' 
) 
parser.add_argument( 
    '--chapters', 
    '--names-list', 
    nargs='*', 
    default=[1,2], 
    help='provide the list of the chapters to be included in the collection (list of strings)' 
) 
parser.add_argument( 
    '--leave_sources', 
    default=True, 
    help='should the script leave all temporary files or not?' 
) 
parser.add_argument( 
    '--hw_mode', 
    default=False, 
    help='in hw mode there is no additional space after examples' 
) 
my_namespace = parser.parse_args() 
 
title    		= my_namespace.title 
chapters 		= my_namespace.chapters 
leave_sources	= my_namespace.leave_sources 
hw_mode			= my_namespace.hw_mode 

title = "msai_5"
chapters = ["Newton method", "Quasi Newton methods"]
 
def find_file_with_title(title, docs): 
	path = os.getcwd() 
	string_to_find = '\ntitle: %s'%title 
	path_to_file = 0 
	for file in docs: 
		with open(file, encoding="utf8") as f: 
			if string_to_find in f.read(): 
				path_to_file = file 
				continue 
 
	if path_to_file == 0: 
		print('ERROR: File with title %s not found'%title) 
	return path_to_file 
 
def delete_header_from_string(string): 
	end_of_header   = string.find('\n---\n') 
	polished_string = string[end_of_header+4:] 
	return polished_string 
 
def increase_header_levels(string): 
	polished_string = string.replace(         '\n##### '	, '\n###### ') 
	polished_string = polished_string.replace('\n#### '		, '\n##### ') 
	polished_string = polished_string.replace('\n### '		, '\n#### ') 
	polished_string = polished_string.replace('\n## '		, '\n### ') 
	polished_string = polished_string.replace('\n# '		, '\n## ') 
	return polished_string 
 
def change_relative_paths_to_absolute(string, path, temporal_path):
	polished_string = string.replace('![](../../', '![](')  
	polished_string = string.replace('![](../', '![](') 
	polished_string = polished_string.replace('src="../', 'src="') 
	for basename in os.listdir(path): 
	    if basename.endswith('.svg') or basename.endswith('.png') or basename.endswith('.jpg') or basename.endswith('.gif') or basename.endswith('.mp4'): 
	        pathname = os.path.join(path, basename) 
	        if os.path.isfile(pathname): 
	            shutil.copy2(pathname, temporal_path) 
	return polished_string 
 
def handle_with_liquid(string): 
	# We start from tabs 
	start_ind 		= 1 
	polished_string = string 
 
	while start_ind != -1: 
		start_ind 		= polished_string.find('{% include tabs.html') 
		if start_ind == -1: 
			continue 
		end_ind 		= polished_string.find('%}') 
		polished_string = polished_string[:start_ind] + polished_string[end_ind+2:] 
 
	# Then, we'll switch to the links 
	polished_string = polished_string.replace('{% include link.html title=\'', '') 
	polished_string = polished_string.replace('\'%}', '') 
 
	# And the last - google colab buttons 
	polished_string = polished_string.replace('[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)]', '[Open in Colab]') 
 
	return polished_string 
 
def catch_problem_from_string(string, path, temporal_path, number): 
	polished_string = change_relative_paths_to_absolute(string, path, temporal_path) 
	polished_string = polished_string.replace('\n1. ', '\nTASK. ', number-1) 
	polished_string = polished_string[polished_string.find('\n1. ')+4:] 
	polished_string = polished_string[:polished_string.find('\n1. ')] 
	polished_string = handle_with_kramdowns_math_list(polished_string) 
	return polished_string 
 
def handle_with_kramdowns_math_list(string): 
	polished_string = string.replace('\n* \$$', '\n* $$')
	polished_string = polished_string.replace('\n1. \$$'	, '\n1. $$')
	polished_string = polished_string.replace('\t* \$$'		, '\t* $$')
	polished_string = polished_string.replace('\t1. \$$'	, '\t1. $$')
	polished_string = polished_string.replace('    * \$$'	, '\t* $$')
	polished_string = polished_string.replace('    1. \$$'	, '\t1. $$')    
	return polished_string
 



# Creating new document and temporary folder 
pdf_folder_name = 'pdfs'
path = os.getcwd()
if not os.path.exists(os.path.join(path, pdf_folder_name)): 
	os.mkdir(os.path.join(path, pdf_folder_name)) 
	print("Temp folder created") 
else: 
	print("Temp folder is already exists") 

path = os.getcwd()
if not os.path.exists(os.path.join(path, pdf_folder_name, title)): 
	os.mkdir(os.path.join(path, pdf_folder_name, title)) 
	print("Folder created") 
else: 
	print("The folder is already exists") 
 
filename = os.path.join(path, pdf_folder_name, title, title+'.md') 
 
main_file = open(filename,"w+", encoding='utf-8') 
# main_file.write("<!-- This file was created by Danya Merkulov's script. See fmin.xyz for more details -->\n") 
# main_file.write("%s\n"%title) 
 
# All documents list 
docs = []  
for root, dirs, files in os.walk(os.path.join(path, 'docs')): 
	dirs[:] = [d for d in dirs if d != 'exercises'] 
	for file in files: 
		if file.endswith(".md"): 
			docs.append(os.path.join(root, file)) 
 
exercises = []  
for root, dirs, files in os.walk(os.path.join(path, 'docs', 'exercises')): 
	for file in files: 
		if file.endswith(".md"): 
			exercises.append(os.path.join(root, file)) 
 
exercises_counter = 0 
for chapter in chapters: 
	# Incuding exercises 
	if chapter.startswith('#ex'): 
		shutil.copy2(os.path.join(path, 'assets', 'images', 'solution.svg'), os.path.join(path, pdf_folder_name, title)) 
		chapter 			= chapter.replace('#ex#', '') 
		numbers 			= chapter[chapter.find('#')+1:].split(',') 
		chapter 			= chapter[:chapter.rfind('#')] 
		chapter_path 		= find_file_with_title(chapter, exercises) 
		chapter_parent_path = chapter_path[:chapter_path.rfind(os.sep)]
		if hw_mode:
			main_file.write('\n# {}\n'.format(chapter)) 
		with open(chapter_path, encoding='utf-8') as f: 
			polished_chapter = f.read() 
			for number in numbers: 
				exercises_counter += 1 
				number 	= int(number) 
				problem = catch_problem_from_string(polished_chapter, chapter_parent_path, os.path.join(path, pdf_folder_name, title), number) 
				if not hw_mode:
					main_file.write('\n\n##### Example {}\n'.format(exercises_counter)) 
					main_file.write(problem) 
					main_file.write('\n![](solution.svg)\n') 
				else:
					main_file.write('\n1. ') 
					main_file.write(problem) 
					# if not main_file.read().endswith('\n'):
					# 	main_file.write('\n') 
		 
	# Including material 
	else: 
		main_file.write('\n# %s\n'%chapter) 
		chapter_path 		= find_file_with_title(chapter, docs) 
		chapter_parent_path = chapter_path[:chapter_path.rfind(os.sep)] 
		with open(chapter_path, encoding='utf-8') as f: 
			polished_chapter = f.read() 
			polished_chapter = delete_header_from_string(polished_chapter) 
			polished_chapter = increase_header_levels(polished_chapter) 
			polished_chapter = change_relative_paths_to_absolute(polished_chapter, chapter_parent_path, os.path.join(path, pdf_folder_name, title)) 
			polished_chapter = handle_with_liquid(polished_chapter) 
			polished_chapter = handle_with_kramdowns_math_list(polished_chapter) 
			main_file.write(polished_chapter) 
 
# Cleaning 
if not leave_sources: 
	 
	shutil.rmtree(os.path.join(path, title)) 
