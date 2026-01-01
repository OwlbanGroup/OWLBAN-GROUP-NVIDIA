import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  ManyToOne,
  CreateDateColumn,
} from 'typeorm';
import { Organization } from '../organizations/organization.entity';

export type BankProvider = 'JPMORGAN';

@Entity('bank_connections')
export class BankConnection {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => Organization, (org) => org.bankConnections)
  organization: Organization;

  @Column({ type: 'varchar' })
  provider: BankProvider;

  @Column()
  providerConnectionId: string; // e.g. JPMorgan reference

  @Column({ default: 'ACTIVE' })
  status: string;

  @CreateDateColumn()
  createdAt: Date;
}
