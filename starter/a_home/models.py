from django.db import models
from django.contrib.auth.models import User

class Proyecto(models.Model):
    nombre = models.CharField(max_length=100)
    descripcion_corta = models.CharField(max_length=255)
    lider_equipo = models.ForeignKey(User, on_delete=models.DO_NOTHING, related_name="proyectos_liderados")
    imagen = models.FileField(upload_to="imagenes")
    url = models.URLField(blank=True, null=True)
    creado_en = models.DateTimeField(auto_now_add=True, verbose_name="Creado en")
    actualizado_en = models.DateTimeField(auto_now=True, verbose_name="Actualizado en")

class Desarrollador(models.Model):
    OPCIONES_CARGO = [
        ('Junior', 'Junior'),
        ('Mid', 'Mid'),
        ('Senior', 'Senior')
    ]
    nombre = models.CharField(max_length=255)
    apellidos = models.CharField(max_length=255)
    NIF = models.CharField(max_length=20)
    email = models.EmailField()
    telefono = models.CharField(max_length=15, null=True)
    github_url = models.URLField(null=True)
    github_username = models.CharField(max_length=50, null=True)
    foto_perfil = models.ImageField(upload_to="perfiles", null=True)
    cargo = models.CharField(choices=OPCIONES_CARGO, max_length=20, default="junior")
    creado_en = models.DateTimeField(auto_now_add=True, verbose_name="Creado en")
    actualizado_en = models.DateTimeField(auto_now=True, verbose_name="Actualizado en")

class Repositorio(models.Model):
    proyecto = models.ForeignKey(Proyecto, on_delete=models.PROTECT, related_name="repositorios")
    url = models.URLField()
    descripcion = models.CharField(max_length=255, blank=True, null=True)
    administrador = models.ForeignKey(User, on_delete=models.PROTECT, related_name="repositorios_administrados")
    creado_en = models.DateTimeField(auto_now_add=True, verbose_name="Creado en")
    actualizado_en = models.DateTimeField(auto_now=True, verbose_name="Actualizado en")

class Servidor(models.Model):
    ip = models.GenericIPAddressField()
    proyecto = models.ForeignKey(Proyecto, on_delete=models.PROTECT, related_name="servidores") 
    administrador = models.ForeignKey(User, on_delete=models.PROTECT, related_name="servidores_administrados")
    usuario_admin = models.CharField(max_length=20, default="root")

class Clave(models.Model):
    OPCIONES_TIPO =[
        ('API key', 'API key'),
        ('Secret', 'Secret'),
        ('Otro', 'Otro')
    ]
    OPCIONES_ENTORNO = [
        ('development', 'development'),
        ('production', 'production'),
        ('staging', 'staging')
    ]
    nombre = models.CharField(max_length=100)
    tipo = models.CharField(choices=OPCIONES_TIPO, max_length=20)
    entorno = models.CharField(choices=OPCIONES_ENTORNO, max_length=20)
    proyecto = models.ForeignKey(Proyecto, on_delete=models.PROTECT, related_name="claves")
