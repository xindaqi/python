from app import db
from sqlalchemy import Column

student_teacher = db.Table('student_teacher', 
    db.Column("student_id", db.Integer, db.ForeignKey("students_info.id")),
    db.Column("teacher_id", db.Integer, db.ForeignKey("teachers_info.id"))
    )


class Student(db.Model):
    __tablename__ = "students_info"
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    id = db.Column(db.Integer, primary_key=True)
    u_id = db.Column(db.String(320), unique=True)
    name = db.Column(db.String(320), unique=False)
    grade = db.Column(db.String(320), unique=False)
    teachers = db.relationship("Teacher", 
    secondary=student_teacher,
    back_populates="students")
    def __repr__(self):
        return "<Student %r>" %self.name

class Teacher(db.Model):
    __tablename__ = "teachers_info"
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    id = db.Column(db.Integer, primary_key=True)
    u_id = db.Column(db.String(320), unique=True)
    name = db.Column(db.String(320), unique=False)
    office = db.Column(db.String(320), unique=False)
    students = db.relationship("Student", 
    secondary=student_teacher,
    back_populates="teachers")