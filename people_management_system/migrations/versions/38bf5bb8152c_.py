"""empty message

Revision ID: 38bf5bb8152c
Revises: e7eba262d455
Create Date: 2019-08-20 10:15:45.708033

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '38bf5bb8152c'
down_revision = 'e7eba262d455'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('people_project', sa.Column('people_co', sa.String(length=320), nullable=True))
    op.drop_constraint('people_project_ibfk_2', 'people_project', type_='foreignkey')
    op.create_foreign_key(None, 'people_project', 'peopleInfo', ['people_co'], ['people_code'])
    op.drop_column('people_project', 'people_code')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('people_project', sa.Column('people_code', mysql.VARCHAR(length=320), nullable=True))
    op.drop_constraint(None, 'people_project', type_='foreignkey')
    op.create_foreign_key('people_project_ibfk_2', 'people_project', 'peopleInfo', ['people_code'], ['people_code'])
    op.drop_column('people_project', 'people_co')
    # ### end Alembic commands ###
