import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  ManyToOne,
  CreateDateColumn,
} from 'typeorm';
import { Organization } from '../organizations/organization.entity';

@Entity('employees')
export class Employee {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => Organization, (org) => org.id)
  organization: Organization;

  @Column()
  name: string;

  @Column()
  email: string;

  @Column()
  bankRoutingNumber: string;

  @Column()
  bankAccountNumber: string;

  @Column('numeric', { precision: 12, scale: 2 })
  payRate: string;

  @Column({ type: 'varchar', default: 'BIWEEKLY' })
  payFrequency: 'WEEKLY' | 'BIWEEKLY' | 'MONTHLY';

  @CreateDateColumn()
  createdAt: Date;
}
