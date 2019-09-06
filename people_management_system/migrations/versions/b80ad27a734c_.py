"""empty message

Revision ID: b80ad27a734c
Revises: 9a518a7db60b
Create Date: 2019-08-20 15:23:30.674244

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b80ad27a734c'
down_revision = '9a518a7db60b'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index('camera_device_code', table_name='camera_device')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_index('camera_device_code', 'camera_device', ['camera_device_code'], unique=True)
    # ### end Alembic commands ###
