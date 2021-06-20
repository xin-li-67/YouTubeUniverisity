import torch

class TextProces:
    def __init__(self):
        char_map_str = """
        ' 0
		<SPACE> 1
		a 2
		b 3
		c 4
		d 5
		e 6
		f 7
		g 8
		h 9
		i 10
		j 11
		k 12
		l 13
		m 14
		n 15
		o 16
		p 17
		q 18
		r 19
		s 20
		t 21
		u 22
		v 23
		w 24
		x 25
		y 26
		z 27
		"""
        self.char_map = {}
        self.index_map = {}

        for line in char_map_str.strip().split('\n'):
            ch, idx = line.split()
            self.char_map[ch] = int(idx)
            self.index_map[int(idx)] = ch
        
        self.index_map[1] = ' '
    
    def text_to_int_sequence(self, text):
        # use a character map and convert text to an integer sequence
        int_seq = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_seq.append(ch)
        
        return int_seq
    
    def int_to_text_sequence(self, labels):
        # use a character map and convert integer labels to an text sequence
        string = []
        for l in labels:
            string.append(self.index_map[l])
        
        return ''.join(string).replace('<SPACE>', ' ')

textprocess = TextProces()

def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []

    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(textprocess.int_to_text_sequence(labels[i][:label_lengths[i]].tolist()))
        for j, idx in enumerate(args):
            if idx != blank_label:
                if collapse_repeated and j != 0 and idx == args[j - 1]:
                    continue
                
                decode.append(idx.item())
        decodes.append(textprocess.int_to_text_sequence(decode))
    
    return decodes, targets