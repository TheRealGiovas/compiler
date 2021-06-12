import string
#Giovanni Esquivel A01633334.
#Fuentes:https://www.youtube.com/watch?v=Eythq9848Fg

gee= 'EE'
gne= 'NE'
glt= 'LT'
ggt= 'GT'
letras = string.ascii_letters
gint= 'INT'
gfloat = 'FLOAT'
gstring	= 'STRING'
gident	= 'IDENTIFIER'
gdivision = 'DIV'
gexp= 'POW'
geq= 'EQ'
glp= 'LPAREN'
grp= 'RPAREN'
gltE= 'LTE'
ggtE= 'GTE'
gc= 'COMMA'
gar	= 'ARROW'
geof= 'EOF'
digitos = '0123456789'
letras_digitos = letras + digitos
gkey= 'KEYWORD'
gmas= 'PLUS'
gmenos= 'MINUS'
gmult = 'MUL'

class Error:
	def __init__(self, ps, pend, error_name, detalle):
		self.ps = ps
		self.pend = pend
		self.error_name = error_name
		self.detalle = detalle
	
	def as_string(self):
		r = f'{self.error_name}: {self.detalle}\n'
		r+= f'File {self.ps.fn}, line {self.ps.ln + 1}'
		r+= '\n\n' + Utility.string_with_arrows(self.ps.ftxt, self.ps, self.pend)
		return result

class IllegalCharError(Error):
	def __init__(self, ps, pend, detalle):
		super().__init__(ps, pend, 'Illegal Character', detalle)

class ExpectedCharError(Error):
	def __init__(self, ps, pend, detalle):
		super().__init__(ps, pend, 'Expected Character', detalle)

class InvalidSyntaxError(Error):
	def __init__(self, ps, pend, detalle=''):
		super().__init__(ps, pend, 'Invalid Syntax', detalle)

class RTError(Error):
	def __init__(self, ps, pend, detalle, context):
		super().__init__(ps, pend, 'executetime Error', detalle)
		self.context = context

	def as_string(self):
		result  = self.generate_traceback()
		result += f'{self.error_name}: {self.detalle}'
		result += '\n\n' + Utility.string_with_arrows(self.ps.ftxt, self.ps, self.pend)
		return result

	def generate_traceback(self):
		result = ''
		pos = self.ps
		ctx = self.context

		while ctx:
			result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
			pos = ctx.parent_entry_pos
			ctx = ctx.parent

		return 'Traceback (most recent call last):\n' + result

class Position:
	def __init__(self, idx, ln, col, fn, ftxt):
		self.idx = idx
		self.ln = ln
		self.col = col
		self.fn = fn
		self.ftxt = ftxt

	def advance(self, current_char=None):
		self.idx += 1
		self.col += 1

		if current_char == '\n':
			self.ln += 1
			self.col = 0

		return self

	def copy(self):
		return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

KEYWORDS = [
	'VAR',
	'AND',
	'OR',
	'NOT',
	'IF',
	'ELIF',
	'ELSE',
	'FOR',
	'TO',
	'STEP',
	'WHILE',
	'THEN'
]
class Token:
	def __init__(self, type_, val=None, ps=None, pend=None):
		self.type = type_
		self.val = val

		if ps:
			self.ps = ps.copy()
			self.pend = ps.copy()
			self.pend.advance()

		if pend:
			self.pend = pend.copy()

	def matches(self, type_, val):
		return self.type == type_ and self.val == val
	
	def __repr__(self):
		if self.val: return f'{self.type}:{self.val}'
		return f'{self.type}'
class Lexer:
	def __init__(self, fn, text):
		self.fn = fn
		self.text = text
		self.pos = Position(-1, 0, -1, fn, text)
		self.current_char = None
		self.advance()	
	def advance(self):
		self.pos.advance(self.current_char)
		self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None
	def make_tokens(self):
		tokens = []
		while self.current_char != None:
			if self.current_char in ' \t':
				self.advance()
			elif self.current_char in digitos:
				tokens.append(self.make_number())
			elif self.current_char in letras:
				tokens.append(self.make_identifier())
			elif self.current_char == '"':
				tokens.append(self.make_string())
			elif self.current_char == '+':
				tokens.append(Token(gmas, ps=self.pos))
				self.advance()
			elif self.current_char == '-':
				tokens.append(self.make_minus_or_arrow())
			elif self.current_char == '*':
				tokens.append(Token(gmult, ps=self.pos))
				self.advance()
			elif self.current_char == '/':
				tokens.append(Token(gdivision, ps=self.pos))
				self.advance()
			elif self.current_char == '^':
				tokens.append(Token(gexp, ps=self.pos))
				self.advance()
			elif self.current_char == '(':
				tokens.append(Token(glp, ps=self.pos))
				self.advance()
			elif self.current_char == ')':
				tokens.append(Token(grp, ps=self.pos))
				self.advance()
			elif self.current_char == '!':
				token, error = self.make_not_equals()
				if error: return [], error
				tokens.append(token)
			elif self.current_char == '=':
				tokens.append(self.make_equals())
			elif self.current_char == '<':
				tokens.append(self.make_less_than())
			elif self.current_char == '>':
				tokens.append(self.make_greater_than())
			elif self.current_char == ',':
				tokens.append(Token(gc, ps=self.pos))
				self.advance()
			else:
				ps = self.pos.copy()
				char = self.current_char
				self.advance()
				return [], IllegalCharError(ps, self.pos, "'" + char + "'")
		tokens.append(Token(geof, ps=self.pos))
		return tokens, None
	def make_number(self):
		num_str = ''
		dot_count = 0
		ps = self.pos.copy()
		while self.current_char != None and self.current_char in digitos + '.':
			if self.current_char == '.':
				if dot_count == 1: break
				dot_count += 1
			num_str += self.current_char
			self.advance()

		if dot_count == 0:
			return Token(gint, int(num_str), ps, self.pos)
		else:
			return Token(gfloat, float(num_str), ps, self.pos)

	def make_string(self):
		string = ''
		ps = self.pos.copy()
		escape_character = False
		self.advance()

		escape_characters = {
			'n': '\n',
			't': '\t'
		}

		while self.current_char != None and (self.current_char != '"' or escape_character):
			if escape_character:
				string += escape_characters.get(self.current_char, self.current_char)
			else:
				if self.current_char == '\\':
					escape_character = True
				else:
					string += self.current_char
			self.advance()
			escape_character = False
		
		self.advance()
		return Token(gstring, string, ps, self.pos)

	def make_identifier(self):
		id_str = ''
		ps = self.pos.copy()

		while self.current_char != None and self.current_char in letras_digitos + '_':
			id_str += self.current_char
			self.advance()

		tok_type = gkey if id_str in KEYWORDS else gident
		return Token(tok_type, id_str, ps, self.pos)

	def make_minus_or_arrow(self):
		tok_type = gmenos
		ps = self.pos.copy()
		self.advance()

		if self.current_char == '>':
			self.advance()
			tok_type = gar

		return Token(tok_type, ps=ps, pend=self.pos)

	def make_not_equals(self):
		ps = self.pos.copy()
		self.advance()

		if self.current_char == '=':
			self.advance()
			return Token(gne, ps=ps, pend=self.pos), None

		self.advance()
		return None, ExpectedCharError(ps, self.pos, "'=' (after '!')")
	
	def make_equals(self):
		tok_type = geq
		ps = self.pos.copy()
		self.advance()

		if self.current_char == '=':
			self.advance()
			tok_type = gee

		return Token(tok_type, ps=ps, pend=self.pos)

	def make_less_than(self):
		tok_type = glt
		ps = self.pos.copy()
		self.advance()

		if self.current_char == '=':
			self.advance()
			tok_type = gltE

		return Token(tok_type, ps=ps, pend=self.pos)

	def make_greater_than(self):
		tok_type = ggt
		ps = self.pos.copy()
		self.advance()

		if self.current_char == '=':
			self.advance()
			tok_type = ggtE

		return Token(tok_type, ps=ps, pend=self.pos)
class NumberNode:
	def __init__(self, tok):
		self.tok = tok

		self.ps = self.tok.ps
		self.pend = self.tok.pend

	def __repr__(self):
		return f'{self.tok}'
class StringNode:
	def __init__(self, tok):
		self.tok = tok

		self.ps = self.tok.ps
		self.pend = self.tok.pend

	def __repr__(self):
		return f'{self.tok}'
class VarAccessNode:
	def __init__(self, var_name_tok):
		self.var_name_tok = var_name_tok

		self.ps = self.var_name_tok.ps
		self.pend = self.var_name_tok.pend
class VarAssignNode:
	def __init__(self, var_name_tok, val_node):
		self.var_name_tok = var_name_tok
		self.val_node = val_node

		self.ps = self.var_name_tok.ps
		self.pend = self.val_node.pend
class BinOpNode:
	def __init__(self, left_node, op_tok, right_node):
		self.left_node = left_node
		self.op_tok = op_tok
		self.right_node = right_node

		self.ps = self.left_node.ps
		self.pend = self.right_node.pend

	def __repr__(self):
		return f'({self.left_node}, {self.op_tok}, {self.right_node})'
class UnaryOpNode:
	def __init__(self, op_tok, node):
		self.op_tok = op_tok
		self.node = node

		self.ps = self.op_tok.ps
		self.pend = node.pend

	def __repr__(self):
		return f'({self.op_tok}, {self.node})'
class IfNode:
	def __init__(self, cases, else_case):
		self.cases = cases
		self.else_case = else_case

		self.ps = self.cases[0][0].ps
		self.pend = (self.else_case or self.cases[len(self.cases) - 1][0]).pend
class ForNode:
	def __init__(self, var_name_tok, start_val_node, end_val_node, step_val_node, body_node):
		self.var_name_tok = var_name_tok
		self.start_val_node = start_val_node
		self.end_val_node = end_val_node
		self.step_val_node = step_val_node
		self.body_node = body_node

		self.ps = self.var_name_tok.ps
		self.pend = self.body_node.pend
class WhileNode:
	def __init__(self, condition_node, body_node):
		self.condition_node = condition_node
		self.body_node = body_node

		self.ps = self.condition_node.ps
		self.pend = self.body_node.pend
class ParseResult:
	def __init__(self):
		self.error = None
		self.node = None
		self.last_registered_advance_count = 0
		self.advance_count = 0

	def register_advancement(self):
		self.last_registered_advance_count = 1
		self.advance_count += 1

	def register(self, res):
		self.last_registered_advance_count = res.advance_count
		self.advance_count += res.advance_count
		if res.error: self.error = res.error
		return res.node

	def success(self, node):
		self.node = node
		return self

	def failure(self, error):
		if not self.error or self.last_registered_advance_count == 0:
			self.error = error
		return self
class Parser:
	def __init__(self, tokens):
		self.tokens = tokens
		self.tok_idx = -1
		self.advance()

	def advance(self, ):
		self.tok_idx += 1
		if self.tok_idx < len(self.tokens):
			self.current_tok = self.tokens[self.tok_idx]
		return self.current_tok

	def parse(self):
		res = self.expr()
		if not res.error and self.current_tok.type != geof:
			return res.failure(InvalidSyntaxError(
				self.current_tok.ps, self.current_tok.pend,
				"Expected '+', '-', '*', '/', '^', '==', '!=', '<', '>', <=', '>=', 'AND' or 'OR'"
			))
		return res

	def expr(self):
		res = ParseResult()

		if self.current_tok.matches(gkey, 'VAR'):
			res.register_advancement()
			self.advance()

			if self.current_tok.type != gident:
				return res.failure(InvalidSyntaxError(
					self.current_tok.ps, self.current_tok.pend,
					"Expected identifier"
				))

			var_name = self.current_tok
			res.register_advancement()
			self.advance()

			if self.current_tok.type != geq:
				return res.failure(InvalidSyntaxError(
					self.current_tok.ps, self.current_tok.pend,
					"Expected '='"
				))

			res.register_advancement()
			self.advance()
			expr = res.register(self.expr())
			if res.error: return res
			return res.success(VarAssignNode(var_name, expr))

		node = res.register(self.bin_op(self.comp_expr, ((gkey, 'AND'), (gkey, 'OR'))))

		if res.error:
			return res.failure(InvalidSyntaxError(
				self.current_tok.ps, self.current_tok.pend,
				"Expected 'VAR', 'IF', 'FOR', 'WHILE', int, float, identifier, '+', '-', '(' or 'NOT'"
			))

		return res.success(node)

	def comp_expr(self):
		res = ParseResult()

		if self.current_tok.matches(gkey, 'NOT'):
			op_tok = self.current_tok
			res.register_advancement()
			self.advance()

			node = res.register(self.comp_expr())
			if res.error: return res
			return res.success(UnaryOpNode(op_tok, node))
		
		node = res.register(self.bin_op(self.arith_expr, (gee, gne, glt, ggt, gltE, ggtE)))
		
		if res.error:
			return res.failure(InvalidSyntaxError(
				self.current_tok.ps, self.current_tok.pend,
				"Expected int, float, identifier, '+', '-', '(' or 'NOT'"
			))

		return res.success(node)

	def arith_expr(self):
		return self.bin_op(self.term, (gmas, gmenos))

	def term(self):
		return self.bin_op(self.factor, (gmult, gdivision))

	def factor(self):
		res = ParseResult()
		tok = self.current_tok

		if tok.type in (gmas, gmenos):
			res.register_advancement()
			self.advance()
			factor = res.register(self.factor())
			if res.error: return res
			return res.success(UnaryOpNode(tok, factor))

		return self.power()

	def power(self):
		return self.bin_op(self.call, (gexp, ), self.factor)

	def call(self):
		res = ParseResult()
		atom = res.register(self.atom())
		if res.error: return res

		if self.current_tok.type == glp:
			res.register_advancement()
			self.advance()
			arg_nodes = []

			if self.current_tok.type == grp:
				res.register_advancement()
				self.advance()
			else:
				arg_nodes.append(res.register(self.expr()))
				if res.error:
					return res.failure(InvalidSyntaxError(
						self.current_tok.ps, self.current_tok.pend,
						"Expected ')', 'VAR', 'IF', 'FOR', 'WHILE',  int, float, identifier, '+', '-', '(' or 'NOT'"
					))

				while self.current_tok.type == gc:
					res.register_advancement()
					self.advance()

					arg_nodes.append(res.register(self.expr()))
					if res.error: return res

				if self.current_tok.type != grp:
					return res.failure(InvalidSyntaxError(
						self.current_tok.ps, self.current_tok.pend,
						f"Expected ',' or ')'"
					))

				res.register_advancement()
				self.advance()
			return res.success(CallNode(atom, arg_nodes))
		return res.success(atom)

	def atom(self):
		res = ParseResult()
		tok = self.current_tok

		if tok.type in (gint, gfloat):
			res.register_advancement()
			self.advance()
			return res.success(NumberNode(tok))

		elif tok.type == gstring:
			res.register_advancement()
			self.advance()
			return res.success(StringNode(tok))

		elif tok.type == gident:
			res.register_advancement()
			self.advance()
			return res.success(VarAccessNode(tok))

		elif tok.type == glp:
			res.register_advancement()
			self.advance()
			expr = res.register(self.expr())
			if res.error: return res
			if self.current_tok.type == grp:
				res.register_advancement()
				self.advance()
				return res.success(expr)
			else:
				return res.failure(InvalidSyntaxError(
					self.current_tok.ps, self.current_tok.pend,
					"Expected ')'"
				))
		
		elif tok.matches(gkey, 'IF'):
			if_expr = res.register(self.if_expr())
			if res.error: return res
			return res.success(if_expr)

		elif tok.matches(gkey, 'FOR'):
			for_expr = res.register(self.for_expr())
			if res.error: return res
			return res.success(for_expr)

		elif tok.matches(gkey, 'WHILE'):
			while_expr = res.register(self.while_expr())
			if res.error: return res
			return res.success(while_expr)

		return res.failure(InvalidSyntaxError(
			tok.ps, tok.pend,
			"Expected int, float, identifier, '+', '-', '(', 'IF', 'FOR', 'WHILE'"
		))

	def if_expr(self):
		res = ParseResult()
		cases = []
		else_case = None

		if not self.current_tok.matches(gkey, 'IF'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.ps, self.current_tok.pend,
				f"Expected 'IF'"
			))

		res.register_advancement()
		self.advance()

		condition = res.register(self.expr())
		if res.error: return res

		if not self.current_tok.matches(gkey, 'THEN'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.ps, self.current_tok.pend,
				f"Expected 'THEN'"
			))

		res.register_advancement()
		self.advance()

		expr = res.register(self.expr())
		if res.error: return res
		cases.append((condition, expr))

		while self.current_tok.matches(gkey, 'ELIF'):
			res.register_advancement()
			self.advance()

			condition = res.register(self.expr())
			if res.error: return res

			if not self.current_tok.matches(gkey, 'THEN'):
				return res.failure(InvalidSyntaxError(
					self.current_tok.ps, self.current_tok.pend,
					f"Expected 'THEN'"
				))

			res.register_advancement()
			self.advance()

			expr = res.register(self.expr())
			if res.error: return res
			cases.append((condition, expr))

		if self.current_tok.matches(gkey, 'ELSE'):
			res.register_advancement()
			self.advance()

			else_case = res.register(self.expr())
			if res.error: return res

		return res.success(IfNode(cases, else_case))

	def for_expr(self):
		res = ParseResult()

		if not self.current_tok.matches(gkey, 'FOR'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.ps, self.current_tok.pend,
				f"Expected 'FOR'"
			))

		res.register_advancement()
		self.advance()

		if self.current_tok.type != gident:
			return res.failure(InvalidSyntaxError(
				self.current_tok.ps, self.current_tok.pend,
				f"Expected identifier"
			))

		var_name = self.current_tok
		res.register_advancement()
		self.advance()

		if self.current_tok.type != geq:
			return res.failure(InvalidSyntaxError(
				self.current_tok.ps, self.current_tok.pend,
				f"Expected '='"
			))
		
		res.register_advancement()
		self.advance()

		start_val = res.register(self.expr())
		if res.error: return res

		if not self.current_tok.matches(gkey, 'TO'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.ps, self.current_tok.pend,
				f"Expected 'TO'"
			))
		
		res.register_advancement()
		self.advance()

		end_val = res.register(self.expr())
		if res.error: return res

		if self.current_tok.matches(gkey, 'STEP'):
			res.register_advancement()
			self.advance()

			step_val = res.register(self.expr())
			if res.error: return res
		else:
			step_val = None

		if not self.current_tok.matches(gkey, 'THEN'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.ps, self.current_tok.pend,
				f"Expected 'THEN'"
			))

		res.register_advancement()
		self.advance()

		body = res.register(self.expr())
		if res.error: return res

		return res.success(ForNode(var_name, start_val, end_val, step_val, body))

	def while_expr(self):
		res = ParseResult()

		if not self.current_tok.matches(gkey, 'WHILE'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.ps, self.current_tok.pend,
				f"Expected 'WHILE'"
			))

		res.register_advancement()
		self.advance()

		condition = res.register(self.expr())
		if res.error: return res

		if not self.current_tok.matches(gkey, 'THEN'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.ps, self.current_tok.pend,
				f"Expected 'THEN'"
			))

		res.register_advancement()
		self.advance()

		body = res.register(self.expr())
		if res.error: return res

		return res.success(WhileNode(condition, body))

	###################################

	def bin_op(self, func_a, ops, func_b=None):
		if func_b == None:
			func_b = func_a
		
		res = ParseResult()
		left = res.register(func_a())
		if res.error: return res

		while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.val) in ops:
			op_tok = self.current_tok
			res.register_advancement()
			self.advance()
			right = res.register(func_b())
			if res.error: return res
			left = BinOpNode(left, op_tok, right)

		return res.success(left)
class RTResult:
	def __init__(self):
		self.val = None
		self.error = None

	def register(self, res):
		self.error = res.error
		return res.val

	def success(self, val):
		self.val = val
		return self

	def failure(self, error):
		self.error = error
		return self
class val:
	def __init__(self):
		self.set_pos()
		self.set_context()

	def set_pos(self, ps=None, pend=None):
		self.ps = ps
		self.pend = pend
		return self

	def set_context(self, context=None):
		self.context = context
		return self

	def added_to(self, other):
		return None, self.illegal_operation(other)

	def subbed_by(self, other):
		return None, self.illegal_operation(other)

	def multed_by(self, other):
		return None, self.illegal_operation(other)

	def dived_by(self, other):
		return None, self.illegal_operation(other)

	def powed_by(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_eq(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_ne(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_lt(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_gt(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_lte(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_gte(self, other):
		return None, self.illegal_operation(other)

	def anded_by(self, other):
		return None, self.illegal_operation(other)

	def ored_by(self, other):
		return None, self.illegal_operation(other)

	def notted(self):
		return None, self.illegal_operation(other)

	def execute(self, args):
		return RTResult().failure(self.illegal_operation())

	def copy(self):
		raise Exception('No copy method defined')

	def is_true(self):
		return False

	def illegal_operation(self, other=None):
		if not other: other = self
		return RTError(
			self.ps, other.pend,
			'Illegal operation',
			self.context
		)
class Number(val):
	def __init__(self, val):
		super().__init__()
		self.val = val

	def added_to(self, other):
		if isinstance(other, Number):
			return Number(self.val + other.val).set_context(self.context), None
		else:
			return None, val.illegal_operation(self, other)

	def subbed_by(self, other):
		if isinstance(other, Number):
			return Number(self.val - other.val).set_context(self.context), None
		else:
			return None, val.illegal_operation(self, other)

	def multed_by(self, other):
		if isinstance(other, Number):
			return Number(self.val * other.val).set_context(self.context), None
		else:
			return None, val.illegal_operation(self, other)

	def dived_by(self, other):
		if isinstance(other, Number):
			if other.val == 0:
				return None, RTError(
					other.ps, other.pend,
					'Division by zero',
					self.context
				)

			return Number(self.val / other.val).set_context(self.context), None
		else:
			return None, val.illegal_operation(self, other)

	def powed_by(self, other):
		if isinstance(other, Number):
			return Number(self.val ** other.val).set_context(self.context), None
		else:
			return None, val.illegal_operation(self, other)

	def get_comparison_eq(self, other):
		if isinstance(other, Number):
			return Number(int(self.val == other.val)).set_context(self.context), None
		else:
			return None, val.illegal_operation(self, other)

	def get_comparison_ne(self, other):
		if isinstance(other, Number):
			return Number(int(self.val != other.val)).set_context(self.context), None
		else:
			return None, val.illegal_operation(self, other)

	def get_comparison_lt(self, other):
		if isinstance(other, Number):
			return Number(int(self.val < other.val)).set_context(self.context), None
		else:
			return None, val.illegal_operation(self, other)

	def get_comparison_gt(self, other):
		if isinstance(other, Number):
			return Number(int(self.val > other.val)).set_context(self.context), None
		else:
			return None, val.illegal_operation(self, other)

	def get_comparison_lte(self, other):
		if isinstance(other, Number):
			return Number(int(self.val <= other.val)).set_context(self.context), None
		else:
			return None, val.illegal_operation(self, other)

	def get_comparison_gte(self, other):
		if isinstance(other, Number):
			return Number(int(self.val >= other.val)).set_context(self.context), None
		else:
			return None, val.illegal_operation(self, other)

	def anded_by(self, other):
		if isinstance(other, Number):
			return Number(int(self.val and other.val)).set_context(self.context), None
		else:
			return None, val.illegal_operation(self, other)

	def ored_by(self, other):
		if isinstance(other, Number):
			return Number(int(self.val or other.val)).set_context(self.context), None
		else:
			return None, val.illegal_operation(self, other)

	def notted(self):
		return Number(1 if self.val == 0 else 0).set_context(self.context), None

	def copy(self):
		copy = Number(self.val)
		copy.set_pos(self.ps, self.pend)
		copy.set_context(self.context)
		return copy

	def is_true(self):
		return self.val != 0
	
	def __repr__(self):
		return str(self.val)
class String(val):
	def __init__(self, val):
		super().__init__()
		self.val = val

	def added_to(self, other):
		if isinstance(other, String):
			return String(self.val + other.val).set_context(self.context), None
		else:
			return None, val.illegal_operation(self, other)

	def multed_by(self, other):
		if isinstance(other, Number):
			return String(self.val * other.val).set_context(self.context), None
		else:
			return None, val.illegal_operation(self, other)

	def is_true(self):
		return len(self.val) > 0

	def copy(self):
		copy = String(self.val)
		copy.set_pos(self.ps, self.pend)
		copy.set_context(self.context)
		return copy

	def __repr__(self):
		return f'"{self.val}"'
class Context:
	def __init__(self, display_name, parent=None, parent_entry_pos=None):
		self.display_name = display_name
		self.parent = parent
		self.parent_entry_pos = parent_entry_pos
		self.symbol_table = None
class tablasim:
	def __init__(self, parent=None):
		self.symbols = {}
		self.parent = parent

	def get(self, name):
		val = self.symbols.get(name, None)
		if val == None and self.parent:
			return self.parent.get(name)
		return val

	def set(self, name, val):
		self.symbols[name] = val

	def remove(self, name):
		del self.symbols[name]
class Interpreter:
	def visit(self, node, context):
		method_name = f'visit_{type(node).__name__}'
		method = getattr(self, method_name, self.no_visit_method)
		return method(node, context)

	def no_visit_method(self, node, context):
		raise Exception(f'No visit_{type(node).__name__} method defined')

	def visit_NumberNode(self, node, context):
		return RTResult().success(
			Number(node.tok.val).set_context(context).set_pos(node.ps, node.pend)
		)

	def visit_StringNode(self, node, context):
		return RTResult().success(
			String(node.tok.val).set_context(context).set_pos(node.ps, node.pend)
		)

	def visit_VarAccessNode(self, node, context):
		res = RTResult()
		var_name = node.var_name_tok.val
		val = context.symbol_table.get(var_name)

		if not val:
			return res.failure(RTError(
				node.ps, node.pend,
				f"'{var_name}' is not defined",
				context
			))

		val = val.copy().set_pos(node.ps, node.pend)
		return res.success(val)

	def visit_VarAssignNode(self, node, context):
		res = RTResult()
		var_name = node.var_name_tok.val
		val = res.register(self.visit(node.val_node, context))
		if res.error: return res

		context.symbol_table.set(var_name, val)
		return res.success(val)

	def visit_BinOpNode(self, node, context):
		res = RTResult()
		left = res.register(self.visit(node.left_node, context))
		if res.error: return res
		right = res.register(self.visit(node.right_node, context))
		if res.error: return res

		if node.op_tok.type == gmas:
			result, error = left.added_to(right)
		elif node.op_tok.type == gmenos:
			result, error = left.subbed_by(right)
		elif node.op_tok.type == gmult:
			result, error = left.multed_by(right)
		elif node.op_tok.type == gdivision:
			result, error = left.dived_by(right)
		elif node.op_tok.type == gexp:
			result, error = left.powed_by(right)
		elif node.op_tok.type == gee:
			result, error = left.get_comparison_eq(right)
		elif node.op_tok.type == gne:
			result, error = left.get_comparison_ne(right)
		elif node.op_tok.type == glt:
			result, error = left.get_comparison_lt(right)
		elif node.op_tok.type == ggt:
			result, error = left.get_comparison_gt(right)
		elif node.op_tok.type == gltE:
			result, error = left.get_comparison_lte(right)
		elif node.op_tok.type == ggtE:
			result, error = left.get_comparison_gte(right)
		elif node.op_tok.matches(gkey, 'AND'):
			result, error = left.anded_by(right)
		elif node.op_tok.matches(gkey, 'OR'):
			result, error = left.ored_by(right)

		if error:
			return res.failure(error)
		else:
			return res.success(result.set_pos(node.ps, node.pend))

	def visit_UnaryOpNode(self, node, context):
		res = RTResult()
		number = res.register(self.visit(node.node, context))
		if res.error: return res

		error = None

		if node.op_tok.type == gmenos:
			number, error = number.multed_by(Number(-1))
		elif node.op_tok.matches(gkey, 'NOT'):
			number, error = number.notted()

		if error:
			return res.failure(error)
		else:
			return res.success(number.set_pos(node.ps, node.pend))

	def visit_IfNode(self, node, context):
		res = RTResult()

		for condition, expr in node.cases:
			condition_val = res.register(self.visit(condition, context))
			if res.error: return res

			if condition_val.is_true():
				expr_val = res.register(self.visit(expr, context))
				if res.error: return res
				return res.success(expr_val)

		if node.else_case:
			else_val = res.register(self.visit(node.else_case, context))
			if res.error: return res
			return res.success(else_val)

		return res.success(None)

	def visit_ForNode(self, node, context):
		res = RTResult()

		start_val = res.register(self.visit(node.start_val_node, context))
		if res.error: return res

		end_val = res.register(self.visit(node.end_val_node, context))
		if res.error: return res

		if node.step_val_node:
			step_val = res.register(self.visit(node.step_val_node, context))
			if res.error: return res
		else:
			step_val = Number(1)

		i = start_val.val

		if step_val.val >= 0:
			condition = lambda: i < end_val.val
		else:
			condition = lambda: i > end_val.val
		
		while condition():
			context.symbol_table.set(node.var_name_tok.val, Number(i))
			i += step_val.val

			res.register(self.visit(node.body_node, context))
			if res.error: return res

		return res.success(None)

	def visit_WhileNode(self, node, context):
		res = RTResult()

		while True:
			condition = res.register(self.visit(node.condition_node, context))
			if res.error: return res

			if not condition.is_true(): break

			res.register(self.visit(node.body_node, context))
			if res.error: return res

		return res.success(None)

		res = RTResult()

		func_name = node.var_name_tok.val if node.var_name_tok else None
		body_node = node.body_node
		arg_names = [arg_name.val for arg_name in node.arg_name_toks]
		func_val = Function(func_name, body_node, arg_names).set_context(context).set_pos(node.ps, node.pend)
		
		if node.var_name_tok:
			context.symbol_table.set(func_name, func_val)

		return res.success(func_val)
class Utility:
	def string_with_arrows(text, ps, pend):
		result = ''
		idx_start = max(text.rfind('\n', 0, ps.idx), 0)
		idx_end = text.find('\n', idx_start + 1)
		if idx_end < 0: idx_end = len(text)
		line_count = pend.ln - ps.ln + 1
		for i in range(line_count):
			line = text[idx_start:idx_end]
			col_start = ps.col if i == 0 else 0
			col_end = pend.col if i == line_count - 1 else len(line) - 1
			result += line + '\n'
			result += ' ' * col_start + '^' * (col_end - col_start)
			idx_start = idx_end
			idx_end = text.find('\n', idx_start + 1)
			if idx_end < 0: idx_end = len(text)
		return result.replace('\t', '')
tabladesimbolosfinal = tablasim()
tabladesimbolosfinal.set("NULL", Number(0))
tabladesimbolosfinal.set("FALSE", Number(0))
tabladesimbolosfinal.set("TRUE", Number(1))

def execute(fn, text):
	# Generate tokens
	lexer = Lexer(fn, text)
	tokens, error = lexer.make_tokens()
	if error: return None, error
	
	# Generate AST
	parser = Parser(tokens)
	ast = parser.parse()
	if ast.error: return None, ast.error

	# execute program
	interpreter = Interpreter()
	context = Context('<program>')
	context.symbol_table = tabladesimbolosfinal
	result = interpreter.visit(ast.node, context)

	return result.val, result.error

if __name__ == "__main__":
	while True:
		text = input('> ')
		result, error = execute('<stdin>', text)
		if error: print(error.as_string())
		elif result: print(result)