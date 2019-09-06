from app import db
from sqlalchemy import Column

people_ironTriangle = db.Table('people_ironTriangle', 
    db.Column("people_code", db.String(320), db.ForeignKey("peopleInfo.people_code")),
    db.Column("ironTriangle_code", db.String(320), db.ForeignKey("ironTriangleInfo.iron_triangle_code"))
    )

people_project = db.Table("people_project",
    db.Column("people_code", db.String(320), db.ForeignKey("peopleInfo.people_code")),
    db.Column("project_code", db.String(320), db.ForeignKey("projectInfo.project_code"))
    )

class People(db.Model):
    __tablename__ = "peopleInfo"
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    id = db.Column(db.Integer, primary_key=True)
    uu_id = db.Column(db.Integer, unique=True)
    people_code = db.Column(db.String(320), unique=True)
    people_name = db.Column(db.String(320), unique=False)
    people_origin = db.Column(db.String(320), unique=False)
    people_sex = db.Column(db.String(320), unique=False)
    people_work_age = db.Column(db.String(320), unique=False)
    people_position = db.Column(db.String(320), unique=False)
    # relation with ironTriangleInfo
    '''
    classname
    middleTable
    relation class variable
    '''
    ironTriangles = db.relationship("IronTriangle", 
    secondary=people_ironTriangle,
    back_populates="peoples")
    # relation with projectInfo
    projects = db.relationship("Project",
    secondary=people_project,
    back_populates="peoples")
    def __repr__(self):
        return "<People %r>" %self.id

class IronTriangle(db.Model):
    __tablename__ = "ironTriangleInfo"
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    id = db.Column(db.Integer, primary_key=True)
    uu_id = db.Column(db.Integer, unique=True)
    iron_triangle_code = db.Column(db.String(320), unique=True)
    iron_triangle_name = db.Column(db.String(320), unique=False)
    # relation with peopleInfo
    peoples = db.relationship("People", 
    secondary=people_ironTriangle,
    back_populates="ironTriangles")
    def __repr__(self):
        return "<IronTriangle %r>" %self.id

class Project(db.Model):
    __tablename__ = "projectInfo"
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    id = db.Column(db.Integer, primary_key=True)
    uu_id = db.Column(db.Integer, unique=True)
    project_code = db.Column(db.String(320), unique=True)
    project_name = db.Column(db.String(320), unique=False)
    project_describe = db.Column(db.String(320), unique=False)
    # relation with peopleInfo
    peoples = db.relationship("People", 
    secondary=people_project,
    back_populates="projects")
    '''
    classname
    tablename
    return query object
    '''
    camera_devices = db.relationship("CameraDevice", backref="projectInfo", lazy="dynamic")

class CameraDevice(db.Model):
    __table_name__ = "cameraDeviceInfo"
    __table_args__ = {"mysql_collate":"utf8_general_ci"}
    id = db.Column(db.Integer, primary_key=True)
    camera_device_code = db.Column(db.String(320), unique=False)
    camera_device_name = db.Column(db.String(320), unique=False)
    camera_device_location = db.Column(db.String(320), unique=False)
    camera_device_describe = db.Column(db.String(320), unique=False)
    project_id = db.Column(db.Integer, db.ForeignKey("projectInfo.uu_id"))
   
