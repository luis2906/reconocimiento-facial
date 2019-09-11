class flabianos:
	""" Lista de invitados"""

	def __init__(self):
		self.Invitados=['luis','juan','pedro','carlos']

	def valida_invitado(self,invitado):
		if invitado in self.Invitados:
			print('Bienvenido {}'.format(invitado))
		else:
			print('Lo siento {}, aun no trais el omnitrix'.format(invitado))