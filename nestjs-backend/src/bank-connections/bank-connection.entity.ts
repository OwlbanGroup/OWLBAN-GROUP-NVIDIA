import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  ManyToOne,
  CreateDateColumn,
  UpdateDateColumn,
} from 'typeorm';
import { Organization } from '../organizations/organization.entity';

export type BankConnectionStatus = 'ACTIVE' | 'INACTIVE' | 'PENDING' | 'ERROR';

@Entity('bank_connections')
export class BankConnection {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  bankName: string;

  @Column({ type: 'varchar' })
  status: BankConnectionStatus;

  @Column({ nullable: true })
  accountNumber?: string;

  @Column({ nullable: true })
  routingNumber?: string;

  @ManyToOne(() => Organization, (org) => org.bankConnections)
  organization: Organization;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;
}
